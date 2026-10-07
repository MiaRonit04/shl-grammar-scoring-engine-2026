"""Extended spectral and temporal functionals from provided recordings only."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import numpy as np,pandas as pd,soundfile as sf,json,hashlib,time
from scipy.signal import stft,resample_poly
from scipy.fft import dct
from math import gcd
ROOT=Path(__file__).resolve().parents[1]
def summarize(x):
 return np.concatenate([np.mean(x,axis=1),np.std(x,axis=1),*np.quantile(x,[.1,.5,.9],axis=1)],axis=0)
def main():
 out=ROOT/'experiments/acoustic_cache';out.mkdir(exist_ok=True);audit=pd.read_csv(ROOT/'artifacts/audio_audit.csv');vecs=[];start=time.time()
 # Fixed 40-band triangular mel bank; no label-dependent fitting.
 mel=lambda hz:2595*np.log10(1+hz/700)
 hz=lambda m:700*(10**(m/2595)-1)
 points=hz(np.linspace(mel(40),mel(7800),42));freq=np.fft.rfftfreq(512,1/16000)
 bank=np.array([np.maximum(0,np.minimum((freq-points[k])/(points[k+1]-points[k]),(points[k+2]-freq)/(points[k+2]-points[k+1]))) for k in range(40)])
 for i,row in audit.iterrows():
  dest=out/(row.sha256+'.npy')
  if dest.exists():v=np.load(dest)
  else:
   a,sr=sf.read(row.path,dtype='float32');a=a.mean(1) if a.ndim>1 else a
   if sr!=16000:a=resample_poly(a,16000//gcd(sr,16000),sr//gcd(sr,16000))
   _,_,z=stft(a,fs=16000,nperseg=400,noverlap=240,nfft=512,boundary=None,padded=False)
   power=np.abs(z)**2;logmel=np.log(np.maximum(bank@power,1e-12));mfcc=dct(logmel,type=2,axis=0,norm='ortho')[:20]
   delta=np.diff(mfcc,axis=1);energy=np.sum(power,axis=0);prob=power/(energy[None,:]+1e-12)
   centroid=np.sum(prob*freq[:,None],axis=0);entropy=-np.sum(prob*np.log(prob+1e-12),axis=0);flat=np.exp(np.mean(np.log(power+1e-12),axis=0))/(power.mean(axis=0)+1e-12)
   dynamics=np.stack([np.log(energy+1e-12),centroid,entropy,flat])
   active=energy>max(np.max(energy)*.01,1e-10)
   v=np.r_[summarize(logmel),summarize(mfcc),summarize(delta),summarize(dynamics),summarize(mfcc[:,active]) if active.any() else np.zeros(100),len(a)/16000,active.mean()].astype('float32')
   assert np.isfinite(v).all();np.save(dest,v)
  vecs.append(v)
  if i%50==0:print(i+1,len(audit),'seconds',round(time.time()-start,1),flush=True)
 np.save(ROOT/'experiments/extended_acoustics.npy',np.vstack(vecs));print('Completed',np.vstack(vecs).shape,flush=True)
if __name__=='__main__':main()
