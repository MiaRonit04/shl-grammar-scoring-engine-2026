"""Pinned W2v-BERT 2.0 frozen features; all recording inference is local."""
import argparse
import hashlib
import json
import os
import time
from math import gcd
from pathlib import Path
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor,Wav2Vec2BertModel

MODEL='facebook/w2v-bert-2.0'
REVISION='da985ba0987f70aaeb84a80f2851cfac8c697a7b'
LAYERS=[6,12,18,24]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--audio-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--float32',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42);torch.set_num_threads(2)
    device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    dtype=torch.float32 if args.float32 or device=='cpu' else torch.float16
    print('Loading',MODEL,REVISION,device,str(dtype),flush=True)
    processor=AutoFeatureExtractor.from_pretrained(MODEL,revision=REVISION)
    model=Wav2Vec2BertModel.from_pretrained(MODEL,revision=REVISION,dtype=dtype,use_safetensors=True).to(device).eval()
    rows=[(split,path) for split in ['train','test'] for path in sorted((args.audio_root/split).glob('*.wav'))]
    assert rows
    if args.limit:rows=rows[:args.limit]
    vectors=[];records=[];start=time.monotonic()
    for split,path in rows:
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        key=hashlib.sha256(f'{sha}|{MODEL}|{REVISION}|{LAYERS}|10s|{dtype}|meanstd|v1'.encode()).hexdigest()
        cache=args.output/(key+'.npy')
        if cache.exists():vector=np.load(cache)
        else:
            audio,sr=sf.read(path,dtype='float32')
            if audio.ndim==2:audio=audio.mean(1)
            if sr!=16000:audio=resample_poly(audio,16000//gcd(sr,16000),sr//gcd(sr,16000))
            count=0;sums=squares=None
            for offset in range(0,len(audio),160000):
                chunk=audio[offset:offset+160000]
                if len(chunk)<640:continue
                batch=processor(chunk,sampling_rate=16000,return_tensors='pt')
                inputs={key:value.to(device=device,dtype=dtype) if value.is_floating_point() else value.to(device)
                        for key,value in batch.items()}
                with torch.inference_mode():
                    hidden=model(**inputs,output_hidden_states=True).hidden_states
                    values=torch.cat([hidden[k][0] for k in LAYERS],dim=-1).float().cpu().numpy().astype(np.float64)
                assert np.isfinite(values).all(),f'Non-finite features: {path.name}; retry with --float32'
                s=values.sum(0);q=np.square(values).sum(0)
                sums=s if sums is None else sums+s;squares=q if squares is None else squares+q;count+=len(values)
            assert count
            mean=sums/count;vector=np.r_[mean,np.sqrt(np.maximum(squares/count-mean**2,0))].astype('float32')
            temp=cache.with_suffix('.tmp')
            with temp.open('wb') as f:np.save(f,vector)
            temp.replace(cache)
        assert vector.shape==(2*len(LAYERS)*model.config.hidden_size,) and np.isfinite(vector).all()
        records.append({'split':split,'filename':path.name,'sha256':sha,'cache_key':key});vectors.append(vector)
        if len(records)==1 or len(records)%10==0:print(f'{len(records)}/{len(rows)} elapsed={time.monotonic()-start:.1f}s',flush=True)
    prefix='pilot' if args.limit else 'complete'
    np.save(args.output/f'{prefix}_embeddings.npy',np.stack(vectors))
    (args.output/f'{prefix}_manifest.json').write_text(json.dumps({
        'model':MODEL,'revision':REVISION,'layers':LAYERS,'hidden_size':model.config.hidden_size,
        'device':device,'dtype':str(dtype),'chunk_seconds':10,'pooling':'frame-weighted mean/std','records':records},indent=2))
    print('Completed',prefix,'elapsed',time.monotonic()-start,flush=True)


if __name__=='__main__':main()
