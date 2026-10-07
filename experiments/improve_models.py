"""Reproducible feature ablations and regression benchmarks; no test labels used."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from catboost import CatBoostRegressor
ROOT=Path(__file__).resolve().parents[1]; A=ROOT/'artifacts'; OUT=ROOT/'experiments/results'; OUT.mkdir(exist_ok=True)
def metrics(y,p):
 return {'rmse':float(np.sqrt(np.mean((y-p)**2))),'pearson':float(np.corrcoef(y,p)[0,1])}
def load():
 folds=pd.read_csv(A/'folds.csv'); y=folds.target.to_numpy(); n=len(y)
 tr=pd.read_csv(A/'train_transcripts.csv'); te=pd.read_csv(A/'test_transcripts.csv'); tx=pd.concat([tr,te],ignore_index=True)
 assert tr.filename.tolist()==folds.filename.tolist()
 meta=tx[['n_segments','avg_logprob','no_speech_prob','segment_gap_mean']].copy()
 meta['empty']=tx.transcript.fillna('').str.strip().eq('').astype(float)
 ling=pd.read_csv(A/'linguistic_features.csv'); audio=pd.read_csv(A/'acoustic_features.csv')
 emb=np.load(A/'text_embeddings.npy')
 basic=pd.concat([ling,meta,audio],axis=1).replace([np.inf,-np.inf],np.nan).to_numpy(float)
 sets={'basic':basic,'audio_meta':pd.concat([audio,meta],axis=1).to_numpy(float),'all':np.c_[basic,emb]}
 if (ROOT/'experiments/extended_acoustics.npy').exists():
  sets['extended']=np.c_[basic,np.load(ROOT/'experiments/extended_acoustics.npy')]
 return folds,y,sets,te

def main():
 folds,y,sets,te=load(); n=len(y); results=[]
 specs=[]
 for feat in ['basic','audio_meta','all']+(['extended'] if 'extended' in sets else []):
  for c,g in [(3,'scale'),(10,'scale'),(30,'scale'),(10,.003)]:
   specs.append((f'{feat}_svr_C{c}_g{g}',feat,SVR(C=c,gamma=g,epsilon=.1)))
 for feat in ['basic','all']+(['extended'] if 'extended' in sets else []):
  for depth,it,l2 in [(3,1200,10),(4,1200,10),(5,1600,10),(6,1600,15)]:
   specs.append((f'{feat}_cat_d{depth}',feat,CatBoostRegressor(iterations=it,depth=depth,learning_rate=.035,l2_leaf_reg=l2,loss_function='RMSE',random_seed=42,verbose=False,thread_count=4,allow_writing_files=False)))
  specs.append((f'{feat}_extra',feat,ExtraTreesRegressor(n_estimators=600,min_samples_leaf=2,max_features=.8,n_jobs=4,random_state=42)))
 for name,feat,est in specs:
  dest=OUT/f'{name}.npz'
  if dest.exists():
   z=np.load(dest); p=z['oof']; test=z['test']
  else:
   p=np.zeros(n); test=[]; start=time.time()
   for f in sorted(folds.fold.unique()):
    va=np.where(folds.fold.to_numpy()==f)[0]; tr=np.where(folds.fold.to_numpy()!=f)[0]
    model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),clone(est))
    model.fit(sets[feat][tr],y[tr]); p[va]=np.clip(model.predict(sets[feat][va]),0,5); test.append(np.clip(model.predict(sets[feat][n:]),0,5))
   test=np.mean(test,axis=0); np.savez(dest,oof=p,test=test)
  result={'model':name,**metrics(y,p),'fold_rmse':[metrics(y[folds.fold==f],p[folds.fold==f])['rmse'] for f in sorted(folds.fold.unique())]}
  results.append(result); print(json.dumps(result),flush=True)
  pd.DataFrame(results).sort_values('rmse').to_csv(OUT/'benchmark.csv',index=False)
if __name__=='__main__':main()
