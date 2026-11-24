import argparse,json,sys,statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'src'))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--benchmark',choices=['screenspot','osworldg','action'],required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--num-shards',type=int,default=1)
    a=p.parse_args()
    if not 0<=a.shard<a.num_shards:p.error('Invalid shard index')
    import torch
    from engine import Engine
    from grounding import parse,correct
    c=json.loads(Path(a.config).read_text());torch.cuda.set_device(0)
    torch.manual_seed(c.get('seed',42));e=Engine(c,0,'joint',a.checkpoint);e.model.eval()
    rows=[json.loads(s) for s in Path(a.data).read_text(encoding='utf-8').splitlines() if s.strip()]
    ids=[str(r['id']) for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate example IDs')
    selected=rows[a.shard::a.num_shards];out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    correct_n=0;times=[];tokens=[];scored=0
    with out.open('x',encoding='utf-8') as f,torch.no_grad():
        for row in selected:
            # Labels stay outside the input passed to inference.
            observable={k:row[k] for k in ['id','image','user','instruction','system'] if k in row}
            rec=e.predict(observable,a.benchmark);rec['id']=row['id']
            if a.benchmark!='action':
                if c.get('format')=='native':
                    from native_prompt import native_action
                    action,error=native_action(rec['raw_response_text']);parsed={'pred_action':action,'parse_error':error}
                else:parsed=parse(rec['raw_response_text'],a.benchmark,'A')
                rec['correct']=bool(correct(row,parsed,a.benchmark));correct_n+=rec['correct'];scored+=1
            times.append(rec['batch_latency_seconds']);tokens.append(rec['generated_tokens'])
            f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush()
    report={'count':len(selected),'correct':correct_n if scored else None,'accuracy':100*correct_n/scored if scored else None,'mean_seconds':statistics.mean(times) if times else None,'mean_output_tokens':statistics.mean(tokens) if tokens else None,'shard':a.shard,'num_shards':a.num_shards}
    out.with_suffix('.summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
if __name__=='__main__':main()
