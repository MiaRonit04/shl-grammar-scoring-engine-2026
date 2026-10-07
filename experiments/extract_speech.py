"""Frozen Wav2Vec2 speech embeddings, computed locally from competition audio only."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(ROOT/'cache/huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import time,json,hashlib
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
from transformers import Wav2Vec2Model
MODEL='facebook/wav2vec2-base-960h'
def bounded_map(pool, fn, rows, limit=8):
 rows=iter(rows);pending=deque()
 for _ in range(limit):
  row=next(rows,None)
  if row is None:break
  pending.append(pool.submit(fn,row))
 while pending:
  yield pending.popleft().result()
  row=next(rows,None)
  if row is not None:pending.append(pool.submit(fn,row))

def main():
 torch.set_num_threads(4)
 device='mps' if torch.backends.mps.is_available() else 'cpu'
 print('Loading',MODEL,'device',device,flush=True)
 torch.manual_seed(42)
 model=Wav2Vec2Model.from_pretrained(MODEL,revision='22aad52d435eb6dbaf354bdad9b0da84ce7d6156',use_safetensors=True).to(device).eval()
 revision=getattr(model.config,'_commit_hash',None)
 out=Path(os.environ.get('GRAMMAR_SPEECH_CACHE',str(ROOT/'experiments/speech_cache')));out.mkdir(exist_ok=True)
 audit=pd.read_csv(ROOT/'artifacts/audio_audit.csv'); start=time.time(); vectors=[]
 def prepare(row):
  key=hashlib.sha256(f'{row.sha256}|{MODEL}|{revision}|chunk10s|layers6,9,12|meanstd|v1'.encode()).hexdigest()
  path=out/f'{key}.npy'
  if path.exists():return path,None
  audio_root=os.environ.get('GRAMMAR_AUDIO_ROOT')
  source=Path(row.path)
  if audio_root:
   staged=Path(audio_root)/row.split/row.filename
   if staged.exists() and hashlib.sha256(staged.read_bytes()).hexdigest()==row.sha256:source=staged
  a,sr=sf.read(source,dtype='float32');a=a.mean(1) if a.ndim>1 else a
  if sr!=16000:a=resample_poly(a,16000//gcd(sr,16000),sr//gcd(sr,16000))
  return path,a
 pool=ThreadPoolExecutor(max_workers=4)
 for i,(path,a) in enumerate(bounded_map(pool,prepare,audit.itertuples())):
  if a is None:vec=np.load(path)
  else:
   sums=None;squares=None;count=0
   for offset in range(0,len(a),160000):
    chunk=a[offset:offset+160000]
    if len(chunk)<400:continue
    chunk=(chunk-chunk.mean())/np.sqrt(chunk.var()+1e-7)
    with torch.inference_mode():
     h=model(torch.from_numpy(chunk).unsqueeze(0).to(device),output_hidden_states=True).hidden_states
     v=torch.cat([h[j][0] for j in [6,9,12]],dim=-1).float().cpu().numpy()
    s=v.sum(0);q=(v*v).sum(0);sums=s if sums is None else sums+s;squares=q if squares is None else squares+q;count+=len(v)
   mean=sums/count;vec=np.r_[mean,np.sqrt(np.maximum(squares/count-mean**2,0))].astype('float32')
   assert np.isfinite(vec).all();np.save(path,vec)
  vectors.append(vec)
  if (i+1)%10==0 or i==0:print(f'{i+1}/{len(audit)} elapsed={time.time()-start:.1f}s',flush=True)
 pool.shutdown()
 np.save(ROOT/'experiments/speech_embeddings.npy',np.vstack(vectors))
 (ROOT/'experiments/speech_manifest.json').write_text(json.dumps({'model':MODEL,'revision':revision,'device':device,'layers':[6,9,12],'chunk_seconds':10,'pooling':'frame-weighted mean and std','filenames':audit.filename.tolist(),'hashes':audit.sha256.tolist()},indent=2))
 print('Completed',flush=True)
if __name__=='__main__':main()
