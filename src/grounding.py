import json,re,math
def formats(arm):
 fields=['action_type','value','point_2d']
 if arm=='B':fields=['action_description','action_target']+fields
 return fields

def prompts(benchmark,arm,observable,definitions):
 # Only these observable keys are read. No target coordinates/refusal labels.
 w,h=observable['image_size'];ins=observable['instruction']
 system=definitions[benchmark]['system']
 start=definitions[benchmark]['user_start'].format(width=w,height=h,instruction=ins)
 schema={}
 if arm=='B':
  schema['action_description']='a short description of the requested operation or refusal reason'
  schema['action_target']='a short description of the target UI element, or None if absent'
 schema.update(action_type='Click or Terminate' if benchmark=='osworldg' else 'Click',
               value='None for Click; a brief refusal reason for Terminate' if benchmark=='osworldg' else 'None',
               point_2d=['x','y'])
 example=json.dumps(schema,ensure_ascii=False,indent=2).replace('"x"','x').replace('"y"','y')
 ending='Coordinates must be normalized to [0, 1000] relative to the full screenshot.'
 if benchmark=='osworldg':ending+=' For Terminate, use point_2d [-100, -100].'
 user=start+'\n\nReturn exactly one valid JSON object between <answer> and </answer>, using these fields in this order:\n<answer>\n'+example+'\n</answer>\n'+ending+'\nReturn only the answer, without a thinking block or other text.'
 return system,user

def point(value):
 if not isinstance(value,list) or len(value)!=2 or any(isinstance(x,bool) for x in value):return None
 try:x,y=map(float,value)
 except (ValueError,TypeError):return None
 if not all(math.isfinite(z) and 0<=z<=1000 for z in (x,y)):return None
 return [x/1000,y/1000]

def payload(raw):
 m=re.search(r'<answer>(.*?)</answer>',raw,re.S|re.I)
 text=m.group(1).strip() if m else raw.strip()
 try:
  x=json.loads(text)
  if isinstance(x,dict):return x
 except (ValueError,TypeError):pass
 decoder=json.JSONDecoder()
 for i,c in enumerate(text):
  if c=='{':
   try:
    x,_=decoder.raw_decode(text[i:])
    if isinstance(x,dict):return x
   except ValueError:pass
 return None

def parse(raw,benchmark,arm,compatible=False):
 native=payload(raw);out=dict(native_action=native,pred_action={},parse_error=None,compatibility_coordinate_field=None)
 thinking=bool(re.search(r'</?(?:thinking|think|thought)>',raw,re.I))
 out['thinking_present']=thinking
 out['schema_fields_match']=isinstance(native,dict) and list(native)==formats(arm)
 if native is None:out['parse_error']='unable_to_parse_answer_json';return out
 kind=re.sub('[^a-z]','',str(native.get('action_type','')).lower())
 out['action_type_valid']=kind in (['click','terminate'] if benchmark=='osworldg' else ['click'])
 if thinking:out['parse_error']='unexpected_visible_reasoning';return out
 if benchmark=='osworldg':
  if kind not in ('click','terminate'):out['parse_error']='unsupported_two_action_output';return out
  if kind=='terminate':
   if native.get('point_2d') not in (None,[-100,-100]):out['parse_error']='refusal_has_coordinates';return out
   out['pred_action']={'type':'wait','status':'failure'};return out
 pt=point(native.get('point_2d'))
 if pt is None and compatible and benchmark=='osworldg' and kind=='click' and 'point_2d' not in native:
  v=native.get('value');field='value'
  if isinstance(v,dict):v=v.get('point_2d');field='value.point_2d'
  if isinstance(v,list) and len(v)==2 and all(isinstance(z,(int,float)) and not isinstance(z,bool) for z in v):
   pt=point(v)
   if pt is not None:out['compatibility_coordinate_field']=field
 if pt is None:out['parse_error']='invalid_point_2d';return out
 # ScreenSpot's primary score is a point hit; action type is recorded separately.
 out['pred_action']={'type':'click','x_norm':pt[0],'y_norm':pt[1]}
 return out

def correct(item,parsed,benchmark):
 a=parsed['pred_action']
 if parsed['parse_error']:return False
 has=a.get('type')=='click' and 'x_norm' in a and 'y_norm' in a
 if benchmark=='osworldg' and item['box_type']=='refusal':return a.get('type') in ('wait','terminate') and not has
 if not has:return False
 w,h=item['image_size'];x=a['x_norm']*w;y=a['y_norm']*h
 if benchmark=='screenspot':
  l,t,r,b=item['bbox_xyxy_original'];return l<=x<=r and t<=y<=b
 if item['box_type']=='bbox':
  l,t,w,h=item['box_coordinates'];return l<=x<=l+w and t<=y<=t+h
 if item['box_type']=='polygon':return point_in_polygon(x,y,item['box_coordinates'])
 raise ValueError('Unknown box type')


def point_in_polygon(x: float, y: float, coordinates: list[float]) -> bool:
    points = list(zip(coordinates[0::2], coordinates[1::2]))
    if len(points) < 3:
        return False
    inside = False
    previous = points[-1]
    for current in points:
        xi, yi = current
        xj, yj = previous
        if (yi > y) != (yj > y):
            crossing_x = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside
