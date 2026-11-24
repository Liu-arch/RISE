import json,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
from latent_core import LatentPolicy,insert_slots
from alignment import pool_historical,hidden_distance,token_kl
from native_prompt import prompt as native_prompt
from grounding import prompts
import action_format

class Engine:
    def __init__(self,config,rank,stage='joint',checkpoint=None):
        self.config=config; self.device=torch.device('cuda',rank)
        model=Qwen3VLForConditionalGeneration.from_pretrained(config['model'],dtype=torch.bfloat16,device_map={'':rank},attn_implementation=config['attention'])
        targets=[n for n,m in model.named_modules() if isinstance(m,nn.Linear) and n.startswith('model.language_model.layers.')]
        if len(targets)!=252:raise ValueError('Expected the 8B Qwen3-VL architecture')
        self.model=LatentPolicy(get_peft_model(model,LoraConfig(task_type='CAUSAL_LM',r=config['lora_r'],lora_alpha=config['lora_alpha'],lora_dropout=config['lora_dropout'],use_rslora=config['lora_use_rslora'],target_modules=targets,bias='none')),8,config['seed'])
        for p in self.model.parameters():
            if p.requires_grad:p.data=p.data.float()
        self.model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        self.model.model.enable_input_require_grads()
        self.processor=AutoProcessor.from_pretrained(config['model'])
        ip=self.processor.image_processor
        ip.min_pixels=config['image_min_pixels'];ip.max_pixels=config['image_max_pixels']
        ip.size={'shortest_edge':ip.min_pixels,'longest_edge':ip.max_pixels}
        if self.processor.tokenizer.pad_token_id is None:self.processor.tokenizer.pad_token=self.processor.tokenizer.eos_token
        if checkpoint:
            p=torch.load(checkpoint,map_location='cpu',weights_only=True)
            set_peft_model_state_dict(self.model.model,p['adapter'])
            with torch.no_grad():self.model.latent_embeddings.copy_(p['latent'])
            actual=get_peft_model_state_dict(self.model.model)
            if set(actual)!=set(p['adapter']) or not all(torch.equal(actual[k].detach().cpu(),v) for k,v in p['adapter'].items()):raise ValueError('Adapter restore mismatch')
    def encode(self,row,benchmark='action'):
        with Image.open(row['image']) as im:
            image=im.convert('RGB');instruction=row.get('user',row.get('instruction'))
            if benchmark=='action':
                if self.config['format']=='native':system=row['system']
                else:system=action_format.system('A')
                user=instruction
            else:
                obs={'instruction':instruction,'image_size':image.size}
                if self.config['format']=='native':system,user=native_prompt(benchmark,'A',obs,None)
                else:
                    defs=json.loads((Path(__file__).parent.parent/'configs/prompts.json').read_text())
                    system,user=prompts(benchmark,'A',obs,defs)
            messages=[{'role':'system','content':system},{'role':'user','content':[{'type':'image'},{'type':'text','text':user}]}]
            text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inp=dict(self.processor(text=[text],images=[image],padding=False,truncation=False,return_tensors='pt'))
        inp.pop('token_type_ids',None)
        return {k:v.to(self.device) for k,v in inp.items()}
    def append(self,inp,ids):
        out=dict(inp);out['input_ids']=torch.cat([inp['input_ids'],torch.tensor([ids],device=self.device,dtype=torch.long)],1);out['attention_mask']=torch.ones_like(out['input_ids']);return out
    def prepare(self,row):
        tok=self.processor.tokenizer;base=self.encode(row);p=base['input_ids'].shape[1]
        target=row['target'] if self.config['format']=='native' else action_format.target(row,'A')[0]
        encoded=tok(target+tok.eos_token,add_special_tokens=False,return_offsets_mapping=True)
        ids=encoded['input_ids'];weights=torch.ones(len(ids),device=self.device)
        if self.config['format']=='action':
            for key,weight in [('action_type',2.),('point_2d',4.)]:
                start=target.index(json.dumps(key)+':')+len(json.dumps(key))+1
                end=start+len(json.dumps(row['action'][key],ensure_ascii=False,separators=(',',':')))
                for i,(s,e) in enumerate(encoded['offset_mapping']):
                    if e>s and s>=start and e<=end:weights[i]=max(float(weights[i]),weight)
        student=insert_slots(self.append(base,ids),p,8,tok.pad_token_id)
        if student['input_ids'].shape[1]>self.config['max_sequence_tokens']:raise ValueError('Training sequence exceeds token budget')
        x={'inputs':student,'labels':torch.tensor(ids,device=self.device),'p':p,'weights':weights,'example_weight':float(row.get('weight',1.0))}
        if self.config['supervision']=='full':
            cot=row['cot'];text='<thinking>'+cot+'</thinking>\n'
            e=tok(text,add_special_tokens=False,return_offsets_mapping=True)
            pos=[p+i for i,(s,t) in enumerate(e['offset_mapping']) if t>s and s>=10 and t<=10+len(cot)]
            if not pos:raise ValueError('Full supervision requires nonempty reasoning')
            x.update(teacher_inputs=self.append(base,e['input_ids']+ids),teacher_positions=pos,teacher_answer=p+len(e['input_ids']))
            if x['teacher_inputs']['input_ids'].shape[1]>self.config['max_sequence_tokens']:raise ValueError('Teacher exceeds token budget')
        return x
    def loss(self,x):
        p=x['p'];labels=x['labels'];weights=x['weights']; teacher=None
        if self.config['supervision']=='full':
            was=self.model.model.training;self.model.model.eval()
            try:
                with torch.no_grad(),self.model.model.disable_adapter():
                    th=self.model.model.get_base_model().model(**x['teacher_inputs'],use_cache=False,return_dict=True).last_hidden_state[0]
                    a=x['teacher_answer'];teacher=(pool_historical(th[x['teacher_positions']],8).detach(),th[a-1:-1].detach(),th[a-1].detach())
            finally:self.model.model.train(was)
        h=self.model.hidden(x['inputs'],p)[0];answer=h[p+7:-1]
        logits=self.model.model.get_output_embeddings()(answer)
        ce=F.cross_entropy(logits.float(),labels,reduction='none')
        loss=(ce*weights).sum()/weights.sum()
        if teacher is not None:
            pooled,ta,tb=teacher
            loss=loss+.1*hidden_distance(h[p:p+8],pooled)+.25*hidden_distance(answer[:1],tb[None])
            kl=loss.new_zeros(())
            for start in range(0,len(labels),32):
                with torch.no_grad():tl=self.model.model.get_output_embeddings()(ta[start:start+32])
                w=weights[start:start+32];kl+=token_kl(logits[start:start+32],tl,w,2.)*w.sum()
            loss=loss+.5*kl/weights.sum()
        return loss*x['example_weight']
    def optimizer(self):return torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],lr=self.config['learning_rate'],weight_decay=self.config['weight_decay'])
    def state(self):return {'adapter':{k:v.detach().cpu() for k,v in get_peft_model_state_dict(self.model.model).items()},'latent':self.model.latent_embeddings.detach().cpu()}
    @torch.no_grad()
    def predict(self,row,benchmark):
        self.model.eval();inp=self.encode(row,benchmark);p=inp['input_ids'].shape[1]
        if p+8>self.config['max_sequence_tokens']:raise ValueError('Input exceeds token budget')
        inp=insert_slots(inp,p,8,self.processor.tokenizer.pad_token_id)
        torch.cuda.synchronize();t=time.perf_counter()
        output=self.model.generate(inp,p,self.config['max_new_tokens']);torch.cuda.synchronize()
        tokens=output[0,p+8:].tolist()
        return {'raw_response_text':self.processor.tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False),'generated_tokens':len(tokens),'batch_latency_seconds':time.perf_counter()-t}
