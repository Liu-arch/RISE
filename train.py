import argparse,json,math,os,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'src'))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--init',help='Initialize trainable model parameters from a checkpoint')
    p.add_argument('--stage',choices=['warmup','joint'],default='joint')
    p.add_argument('--epochs',type=int,default=1);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--save-every',type=int,default=500)
    a=p.parse_args()
    if min(a.epochs,a.batch_size,a.save_every)<1:p.error('epochs, batch size and save interval must be positive')
    import torch
    import torch.distributed as dist
    from engine import Engine
    c=json.loads(Path(a.config).read_text());rank=int(os.environ.get('LOCAL_RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
    torch.cuda.set_device(rank);torch.manual_seed(c.get('seed',42));random.seed(c.get('seed',42))
    if world>1:dist.init_process_group('nccl')
    out=Path(a.output)
    if rank==0:out.mkdir(parents=True,exist_ok=False)
    if world>1:dist.barrier()
    rows=[json.loads(s) for s in Path(a.data).read_text(encoding='utf-8').splitlines() if s.strip()]
    if not rows:raise ValueError('Empty training data')
    e=Engine(c,rank,a.stage,a.init);e.model.train();opt=e.optimizer()
    class LossModule(torch.nn.Module):
        def __init__(self):super().__init__();self.model=e.model
        def forward(self,x):return e.loss(x)
    mod=LossModule()
    if world>1:mod=torch.nn.parallel.DistributedDataParallel(mod,device_ids=[rank],find_unused_parameters=True)
    total=a.epochs*math.ceil(len(rows)/a.batch_size);step=0;base_lrs=[g['lr'] for g in opt.param_groups]
    def cpu(x):
        if torch.is_tensor(x):return x.detach().cpu()
        if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
        if isinstance(x,list):return [cpu(v) for v in x]
        if isinstance(x,tuple):return tuple(cpu(v) for v in x)
        return x
    try:
        for epoch in range(a.epochs):
            order=list(range(len(rows)));random.Random(c.get('seed',42)+epoch).shuffle(order)
            for start in range(0,len(order),a.batch_size):
                group=order[start:start+a.batch_size];n=len(group);opt.zero_grad(set_to_none=True)
                ratio=min(1.,(step+1)/max(1,total*.03))*(.1+.9*.5*(1+math.cos(math.pi*step/max(1,total-1))))
                for g,lr in zip(opt.param_groups,base_lrs):g['lr']=lr*ratio
                value=torch.zeros((),device=rank)
                for j in range(math.ceil(n/world)):
                    pos=j*world+rank;valid=pos<n;sample=e.prepare(rows[group[pos if valid else 0]])
                    with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):loss=mod(sample)
                    if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
                    (loss*(world/n if valid else 0.)).backward()
                    if valid:value+=loss.detach()/n
                norm=torch.nn.utils.clip_grad_norm_(e.model.parameters(),1.)
                if not torch.isfinite(norm):raise RuntimeError('Non-finite gradient')
                opt.step();step+=1
                if world>1:dist.all_reduce(value)
                if rank==0:
                    record={'step':step,'total':total,'loss':float(value)}
                    print(json.dumps(record),flush=True)
                    with (out/'train.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                    if step%a.save_every==0 or step==total:
                        checkpoint={'config':c,'stage':a.stage,'step':step,'epoch':epoch,'cursor':start+n,'world_size':world}
                        torch.save({**e.state(),'optimizer':cpu(opt.state_dict()),'metadata':checkpoint},out/f'checkpoint-{step:06d}.pt')
                if world>1:dist.barrier()
    finally:
        if dist.is_initialized():dist.destroy_process_group()
if __name__=='__main__':main()
