"""Shared prompts and deterministic action-only/description-action label views."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
COMMON_SYSTEM='''You are a GUI agent. Given the current screenshot, the user's instruction, and previous actions, predict the next single action.
Available actions:
Click: click a UI element; point_2d gives its location.
LongPress: long-press a UI element; point_2d gives its location.
Write: enter value as text, at point_2d when provided or at the current focus otherwise.
Select: select the option specified by value, at point_2d when provided.
KeyboardPress: press the key or key combination specified by value.
Scroll: scroll in the direction specified by value.
Swipe: swipe in the direction specified by value, at point_2d when provided.
Wait: wait for the number of seconds specified by value.
NavigateHome: go to the device home screen.
NavigateBack: go back.
OpenApp: open the application specified by value.
Answer: answer the user's question using value.
Terminate: end the completed task, with a message in value when available.
Coordinates use [x,y] normalized to [0,1000] relative to the full original screenshot. Preserve meaningful text, key, direction, application, and other operation arguments in value. Use the string "None" for absent values or coordinates.
Return exactly one <answer>...</answer> containing a JSON object. Do not output a thinking block or text outside the answer.'''
FORMATS={
'A':'Output exactly these fields in this order: action_type, value, point_2d.',
'B':'Output exactly these fields in this order: action_description, action_target, action_type, value, point_2d. action_description briefly states the next operation. action_target describes the target UI element, or is "None" when not applicable.'
}
def system(arm):return COMMON_SYSTEM+'\n'+FORMATS[arm]
def messages(row,arm):
 # Do not pass the row dict or any current gold fields as prompt data.
 return [{'role':'system','content':system(arm)},
         {'role':'user','content':[{'type':'image'},{'type':'text','text':row['user']}]}]
def target(row,arm):
 a=row['action'];core=('action_type','value','point_2d')
 fields=core if arm=='A' else ('action_description','action_target',*core)
 text='<answer>{'
 for i,name in enumerate(fields):
  if i:text+=','
  text+=json.dumps(name)+':'+json.dumps(a[name],ensure_ascii=False,separators=(',',':'))
 text+='}</answer>'
 return text,(len('<answer>{'),text.index('"action_type"')) if arm=='B' else None
def target_encoding(tokenizer,row,arm):
 text,aux=target(row,arm)
 e=tokenizer(text+tokenizer.eos_token,add_special_tokens=False,return_offsets_mapping=True)
 # Boundary syntax belongs to action/format CE; only the auxiliary field span
 # has its own normalization. Descriptions cannot dilute executable CE.
 groups=[2 if aux and s>=aux[0] and t<=aux[1] and t>s else 1 for s,t in e['offset_mapping']]
 assert 1 in groups and (arm=='A' or 2 in groups)
 return e['input_ids'],groups,text
def processor(config):
 from transformers import AutoProcessor
 p=AutoProcessor.from_pretrained(config['model'],local_files_only=False)
 ip=p.image_processor
 ip.min_pixels=config['image_min_pixels'];ip.max_pixels=config['image_max_pixels']
 ip.size={'shortest_edge':config['image_min_pixels'],'longest_edge':config['image_max_pixels']}
 p.tokenizer.padding_side='right'
 return p
def prefix(p,row,arm):return p.apply_chat_template(messages(row,arm),tokenize=False,add_generation_prompt=True)
