"""Compare fold-fitted PCA compression of the two strongest speech encoders."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from evaluate_representation import metrics


def aligned_layer(folder, layer, folds, test, hashes):
    manifest = json.loads((folder/'complete_manifest.json').read_text())
    raw = np.load(folder/'complete_embeddings.npy')
    records = manifest['records']
    index = {(r['split'], r['filename']): i for i, r in enumerate(records)}
    assert len(index) == len(records) == len(raw)
    assert all(hashes[(r['split'], r['filename'])] == r['sha256'] for r in records)
    order = [index[('train', name)] for name in folds.filename]+[index[('test', name)] for name in test.filename]
    d = manifest['hidden_size']
    k = manifest['layers'].index(layer)
    total = d*len(manifest['layers'])
    return np.c_[raw[order, k*d:(k+1)*d], raw[order, total+k*d:total+(k+1)*d]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wavlm', type=Path, required=True)
    p.add_argument('--whisper', type=Path, required=True)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    folds = pd.read_csv(args.metadata/'folds.csv')
    test = pd.read_csv(args.metadata/'test_transcripts.csv')
    audit = pd.read_csv(args.metadata/'audio_audit.csv')
    hashes = {(r.split, r.filename):r.sha256 for r in audit.itertuples()}
    matrices = dict(wavlm=aligned_layer(args.wavlm, 12, folds, test, hashes),
                    whisper=aligned_layer(args.whisper, 32, folds, test, hashes))
    y = folds.target.to_numpy()
    n = len(y)
    split = [(np.flatnonzero(folds.fold != f), np.flatnonzero(folds.fold == f)) for f in sorted(folds.fold.unique())]
    fingerprint = hashlib.sha256(Path(__file__).read_bytes()+y.tobytes()+folds.fold.to_numpy().tobytes()
                                 +b''.join(a.tobytes() for a in matrices.values())).hexdigest()
    specs = {}
    for name, x in matrices.items():
        for components in (64,128,256):
            for c in (10,30):
                specs[f'{name}_pca{components}_svr{c}'] = (x, make_pipeline(
                    StandardScaler(), PCA(n_components=components, svd_solver='randomized', random_state=42),
                    SVR(C=c, gamma='scale', epsilon=.1)))
    results = []
    for name, (x, estimator) in specs.items():
        dest = args.output/(name+'.npz')
        if dest.exists():
            z = np.load(dest)
            assert str(z['fingerprint']) == fingerprint
            oof = z['oof']
        else:
            oof = np.zeros(n)
            bagged = []
            for tr, va in split:
                assert not set(folds.audio_hash.iloc[tr]) & set(folds.audio_hash.iloc[va])
                model = clone(estimator).fit(x[tr], y[tr])
                oof[va] = np.clip(model.predict(x[va]), 0, 5)
                bagged.append(np.clip(model.predict(x[n:]), 0, 5))
            np.savez(dest, oof=oof, test=np.mean(bagged, axis=0), fingerprint=fingerprint)
        row = dict(model=name, **metrics(y,oof), folds=[metrics(y[va],oof[va]) for _,va in split])
        results.append(row)
        print(json.dumps(row), flush=True)
        (args.output/'benchmark.json').write_text(json.dumps(sorted(results,key=lambda r:r['rmse']), indent=2))
    best = min(results,key=lambda r:r['rmse'])
    x, estimator = specs[best['model']]
    secondary = np.zeros(n)
    bins = pd.qcut(y,5,labels=False,duplicates='drop')
    for tr,va in StratifiedGroupKFold(5,shuffle=True,random_state=2026).split(x[:n],bins,folds.audio_hash):
        secondary[va] = np.clip(clone(estimator).fit(x[tr],y[tr]).predict(x[va]),0,5)
    model = clone(estimator).fit(x[:n],y)
    train, full_test = np.clip(model.predict(x[:n]),0,5),np.clip(model.predict(x[n:]),0,5)
    selected = np.load(args.output/(best['model']+'.npz'))
    np.savez(args.output/'selected.npz',oof=selected['oof'],secondary_oof=secondary,
             train=train,full_test=full_test,bagged_test=selected['test'])
    summary = dict(selected=best,secondary=metrics(y,secondary),training=metrics(y,train),
                   fingerprint=fingerprint,method='Fold-fitted StandardScaler/PCA/SVR',
                   limitation='Development selection and secondary split reuse labels; neither is an untouched holdout.')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
