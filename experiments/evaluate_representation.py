"""Evaluate a cached frozen speech representation on fixed, grouped folds."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold


def metrics(y, pred):
    return {'rmse': float(np.sqrt(np.mean((y-pred)**2))),
            'pearson': float(np.corrcoef(y, pred)[0, 1])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pooling', choices=['meanstd', 'mean'], default='meanstd')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    folds = pd.read_csv(args.metadata/'folds.csv')
    test = pd.read_csv(args.metadata/'test_transcripts.csv')
    manifest = json.loads((args.features/'complete_manifest.json').read_text())
    raw = np.load(args.features/'complete_embeddings.npy')
    records = manifest['records']
    index = {(r['split'], r['filename']): i for i, r in enumerate(records)}
    assert len(index) == len(records) == len(raw)
    order = [index[('train', f)] for f in folds.filename] + [index[('test', f)] for f in test.filename]
    audit = pd.read_csv(args.metadata/'audio_audit.csv')
    hashes = {(r.split, r.filename): r.sha256 for r in audit.itertuples()}
    assert all(hashes[(r['split'], r['filename'])] == r['sha256'] for r in records)
    raw = raw[order]
    assert np.isfinite(raw).all()
    n = len(folds)
    y = folds.target.to_numpy()
    dim = manifest['hidden_size']
    layers = manifest['layers']
    total = dim * len(layers)
    matrices = {str(layer): np.c_[raw[:, k*dim:(k+1)*dim], raw[:, total+k*dim:total+(k+1)*dim]]
                for k, layer in enumerate(layers)}
    if args.pooling == 'mean':
        matrices = {layer: x[:, :dim] for layer, x in matrices.items()}
    specs = {}
    for layer, x in matrices.items():
        grid = [(3, 1), (10, 1)] if args.pooling == 'mean' else [(3, 1), (10, 1), (30, .25), (100, .25)]
        for c, multiplier in grid:
            specs[f'layer{layer}_svr{c}_g{multiplier}'] = (
                x, make_pipeline(StandardScaler(), SVR(C=c, gamma=multiplier/x.shape[1], epsilon=.1)))
        if args.pooling == 'meanstd':
            specs[f'layer{layer}_ridge1000'] = (x, make_pipeline(StandardScaler(), Ridge(alpha=1000)))
    split = [(np.flatnonzero(folds.fold != f), np.flatnonzero(folds.fold == f)) for f in sorted(folds.fold.unique())]
    fingerprint = hashlib.sha256(raw.tobytes()+y.tobytes()+folds.fold.to_numpy().tobytes()+args.pooling.encode()).hexdigest()
    results = []
    for name, (x, estimator) in specs.items():
        dest = args.output/(name+'.npz')
        if dest.exists():
            with np.load(dest) as cache:
                assert str(cache['fingerprint']) == fingerprint
                oof = cache['oof']
        else:
            oof = np.zeros(n)
            predictions = []
            for tr, va in split:
                model = clone(estimator).fit(x[tr], y[tr])
                oof[va] = np.clip(model.predict(x[va]), 0, 5)
                predictions.append(np.clip(model.predict(x[n:]), 0, 5))
            np.savez(dest, oof=oof, test=np.mean(predictions, axis=0), fingerprint=fingerprint)
        row = {'model': name, **metrics(y, oof), 'folds': [metrics(y[va], oof[va]) for _, va in split]}
        results.append(row)
        print(json.dumps(row), flush=True)
        (args.output/'benchmark.json').write_text(json.dumps(sorted(results, key=lambda r:r['rmse']), indent=2))
    best = min(results, key=lambda r:r['rmse'])
    name = best['model']
    x, estimator = specs[name]
    secondary = np.zeros(n)
    bins = pd.qcut(y, 5, labels=False, duplicates='drop')
    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=2026).split(x[:n], bins, folds.audio_hash):
        secondary[va] = np.clip(clone(estimator).fit(x[tr], y[tr]).predict(x[va]), 0, 5)
    model = clone(estimator).fit(x[:n], y)
    train = np.clip(model.predict(x[:n]), 0, 5)
    full_test = np.clip(model.predict(x[n:]), 0, 5)
    np.savez(args.output/'selected.npz', secondary_oof=secondary, train=train, full_test=full_test,
             oof=np.load(args.output/(name+'.npz'))['oof'],
             bagged_test=np.load(args.output/(name+'.npz'))['test'])
    summary = {'selected': best, 'secondary': metrics(y, secondary), 'training': metrics(y, train),
               'model': manifest['model'], 'revision': manifest['revision'], 'fingerprint': fingerprint, 'pooling': args.pooling,
               'limitation': 'Development selection and secondary split reuse labels; neither is an untouched holdout.'}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
