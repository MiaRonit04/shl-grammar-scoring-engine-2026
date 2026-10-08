"""Fold-safe joint regression over complementary frozen speech encoders."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.model_selection import StratifiedGroupKFold
from evaluate_representation import metrics


def balance(x, widths):
    out=x.copy()
    start=0
    for width in widths:
        out[:,start:start+width]/=np.sqrt(width)
        start+=width
    return out


def representation(path,layer,filenames,hashes):
    manifest=json.loads((path/'complete_manifest.json').read_text())
    records=manifest['records']
    index={(r['split'],r['filename']):i for i,r in enumerate(records)}
    order=[index[key] for key in filenames]
    assert all(records[i]['sha256']==sha for i,sha in zip(order,hashes))
    raw=np.load(path/'complete_embeddings.npy')[order]
    d=manifest['hidden_size'];k=manifest['layers'].index(layer);total=d*len(manifest['layers'])
    return np.c_[raw[:,k*d:(k+1)*d],raw[:,total+k*d:total+(k+1)*d]]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--wavlm',type=Path,required=True)
    p.add_argument('--whisper',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    folds=pd.read_csv(args.metadata/'folds.csv');test=pd.read_csv(args.metadata/'test_transcripts.csv')
    n=len(folds);y=folds.target.to_numpy()
    names=[('train',f) for f in folds.filename]+[('test',f) for f in test.filename]
    hashes=folds.audio_hash.tolist()+test.audio_hash.tolist()
    a=representation(args.wavlm,12,names,hashes)
    b=representation(args.whisper,32,names,hashes)
    basic=np.load(args.metadata/'extended.npy')
    sets={'speech':(np.c_[a,b],[a.shape[1],b.shape[1]]),
          'speech_handcrafted':(np.c_[a,b,basic],[a.shape[1],b.shape[1],basic.shape[1]])}
    specs={};results=[]
    for feature,(x,widths) in sets.items():
        for c in [3,10,30]:
            name=f'{feature}_svr{c}'
            est=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),
                              FunctionTransformer(balance,kw_args={'widths':widths}),SVR(C=c,gamma=1/len(widths),epsilon=.1))
            specs[name]=(x,est)
            oof=np.zeros(n);tp=[]
            for f in sorted(folds.fold.unique()):
                tr=np.flatnonzero(folds.fold.ne(f));va=np.flatnonzero(folds.fold.eq(f))
                model=clone(est).fit(x[tr],y[tr])
                oof[va]=np.clip(model.predict(x[va]),0,5);tp.append(np.clip(model.predict(x[n:]),0,5))
            np.savez(args.output/(name+'.npz'),oof=oof,test=np.mean(tp,axis=0))
            row={'model':name,**metrics(y,oof)};results.append(row);print(json.dumps(row),flush=True)
            (args.output/'benchmark.json').write_text(json.dumps(results,indent=2))
    best=min(results,key=lambda r:r['rmse']);x,est=specs[best['model']]
    secondary=np.zeros(n);bins=pd.qcut(y,5,labels=False,duplicates='drop')
    for tr,va in StratifiedGroupKFold(5,shuffle=True,random_state=2026).split(x[:n],bins,folds.audio_hash):
        secondary[va]=np.clip(clone(est).fit(x[tr],y[tr]).predict(x[va]),0,5)
    model=clone(est).fit(x[:n],y);train=np.clip(model.predict(x[:n]),0,5)
    z=np.load(args.output/(best['model']+'.npz'))
    np.savez(args.output/'selected.npz',oof=z['oof'],bagged_test=z['test'],secondary_oof=secondary,
             train=train,full_test=np.clip(model.predict(x[n:]),0,5))
    summary={'selected':best,'secondary':metrics(y,secondary),'training':metrics(y,train),
             'method':'WavLM layer12 and Whisper layer32 mean/std; fold-fitted standardization and block scaling; SVR'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
