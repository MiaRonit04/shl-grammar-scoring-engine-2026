"""Compare squared-error kernel regression with the existing speech SVRs."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from sklearn.kernel_ridge import KernelRidge
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import StratifiedGroupKFold
from evaluate_joint_speech import representation,balance
from evaluate_representation import metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--wavlm',type=Path,required=True)
    parser.add_argument('--whisper',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    folds=pd.read_csv(args.metadata/'folds.csv');test=pd.read_csv(args.metadata/'test_transcripts.csv')
    n=len(folds);y=folds.target.to_numpy()
    names=[('train',f) for f in folds.filename]+[('test',f) for f in test.filename]
    hashes=folds.audio_hash.tolist()+test.audio_hash.tolist()
    a=representation(args.wavlm,12,names,hashes);b=representation(args.whisper,32,names,hashes)
    sets={'wavlm':(a,[a.shape[1]],1),'whisper':(b,[b.shape[1]],.25),
          'joint':(np.c_[a,b],[a.shape[1],b.shape[1]],.5)}
    specs={};results=[]
    for tag,(x,widths,gamma) in sets.items():
        for alpha in [.01,.1,1.0]:
            name=f'{tag}_krr{alpha}'
            estimator=TransformedTargetRegressor(regressor=make_pipeline(
                StandardScaler(),FunctionTransformer(balance,kw_args={'widths':widths}),
                KernelRidge(alpha=alpha,kernel='rbf',gamma=gamma)),transformer=StandardScaler())
            specs[name]=(x,estimator)
            oof=np.zeros(n);tp=[]
            for f in sorted(folds.fold.unique()):
                tr=np.flatnonzero(folds.fold.ne(f));va=np.flatnonzero(folds.fold.eq(f))
                fitted=clone(estimator).fit(x[tr],y[tr])
                oof[va]=np.clip(fitted.predict(x[va]),0,5);tp.append(np.clip(fitted.predict(x[n:]),0,5))
            np.savez(args.output/(name+'.npz'),oof=oof,test=np.mean(tp,axis=0))
            row={'model':name,**metrics(y,oof)};results.append(row);print(json.dumps(row),flush=True)
            (args.output/'benchmark.json').write_text(json.dumps(results,indent=2))
    best=min(results,key=lambda r:r['rmse']);x,estimator=specs[best['model']]
    secondary=np.zeros(n);bins=pd.qcut(y,5,labels=False,duplicates='drop')
    for tr,va in StratifiedGroupKFold(5,shuffle=True,random_state=2026).split(x[:n],bins,folds.audio_hash):
        secondary[va]=np.clip(clone(estimator).fit(x[tr],y[tr]).predict(x[va]),0,5)
    fitted=clone(estimator).fit(x[:n],y);train=np.clip(fitted.predict(x[:n]),0,5)
    cached=np.load(args.output/(best['model']+'.npz'))
    np.savez(args.output/'selected.npz',oof=cached['oof'],bagged_test=cached['test'],secondary_oof=secondary,
             train=train,full_test=np.clip(fitted.predict(x[n:]),0,5))
    summary={'selected':best,'secondary':metrics(y,secondary),'training':metrics(y,train),
             'method':'RBF kernel ridge with fold-fitted input scaling and target centering/scaling'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
