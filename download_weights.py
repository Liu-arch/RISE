import argparse,hashlib
from huggingface_hub import hf_hub_download

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='weights');a=p.parse_args()
    file=hf_hub_download('L1007555652/RISE','RISE.pt',local_dir=a.output)
    h=hashlib.sha256()
    with open(file,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    if h.hexdigest()!='e84916b1bfca8384e6affb12b9808452812076743f9e192843fac25106c58027':raise RuntimeError('Checkpoint checksum mismatch')
    print(file)
if __name__=='__main__':main()
