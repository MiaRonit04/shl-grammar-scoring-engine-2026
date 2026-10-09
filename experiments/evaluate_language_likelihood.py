"""Ablate language likelihood features alongside the existing handcrafted model."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from evaluate_representation import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--language', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--feature-file', default='likelihood.npy')
    parser.add_argument('--feature-name', default='likelihood')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    folds = pd.read_csv(args.metadata/'folds.csv')
    test = pd.read_csv(args.metadata/'test_transcripts.csv')
    n = len(folds)
    y = folds.target.to_numpy()
    manifest = json.loads((args.language/'complete_manifest.json').read_text())
    assert [r['filename'] for r in manifest['records']] == folds.filename.tolist()+test.filename.tolist()
    assert [r['split'] for r in manifest['records']] == ['train']*n+['test']*len(test)
    audit = pd.read_csv(args.metadata/'audio_audit.csv')
    hashes = {(r.split,r.filename):r.sha256 for r in audit.itertuples()}
    assert all(hashes[(r['split'],r['filename'])] == r['sha256'] for r in manifest['records'])
    likelihood = np.load(args.language/args.feature_file)
    extended = np.load(args.metadata/'extended.npy')
    assert len(likelihood) == len(extended) == n+len(test)
    assert np.isfinite(likelihood).all()
    fingerprint = hashlib.sha256(Path(__file__).read_bytes()+likelihood.tobytes()+extended.tobytes()
                                 +y.tobytes()+folds.fold.to_numpy().tobytes()).hexdigest()
    sets = {args.feature_name: likelihood, 'extended_'+args.feature_name: np.c_[extended, likelihood]}
    models = {}
    rows = []
    for name, x in sets.items():
        for depth in ([3] if name == args.feature_name else [4,5]):
            key = f'{name}_cat{depth}'
            est = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), CatBoostRegressor(
                iterations=1200 if depth < 5 else 1600, depth=depth, learning_rate=.035,
                l2_leaf_reg=10, loss_function='RMSE', random_seed=42, verbose=False,
                thread_count=2, allow_writing_files=False))
            models[key] = (x, est)
            dest = args.output/(key+'.npz')
            if dest.exists():
                cache = np.load(dest)
                assert str(cache['fingerprint']) == fingerprint
                oof = cache['oof']
            else:
                oof = np.zeros(n)
                predictions = []
                for f in sorted(folds.fold.unique()):
                    tr = np.flatnonzero(folds.fold != f)
                    va = np.flatnonzero(folds.fold == f)
                    assert not set(folds.audio_hash.iloc[tr]) & set(folds.audio_hash.iloc[va])
                    fitted = clone(est).fit(x[tr], y[tr])
                    oof[va] = np.clip(fitted.predict(x[va]), 0, 5)
                    predictions.append(np.clip(fitted.predict(x[n:]), 0, 5))
                np.savez(dest, oof=oof, test=np.mean(predictions, axis=0),fingerprint=fingerprint)
            row = {'model': key, **metrics(y, oof)}
            rows.append(row)
            print(json.dumps(row), flush=True)
            (args.output/'benchmark.json').write_text(json.dumps(rows, indent=2))
    best = min(rows, key=lambda r:r['rmse'])
    x, est = models[best['model']]
    secondary = np.zeros(n)
    bins = pd.qcut(y, 5, labels=False, duplicates='drop')
    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=2026).split(x[:n], bins, folds.audio_hash):
        secondary[va] = np.clip(clone(est).fit(x[tr], y[tr]).predict(x[va]), 0, 5)
    fitted = clone(est).fit(x[:n], y)
    train = np.clip(fitted.predict(x[:n]), 0, 5)
    cache = np.load(args.output/(best['model']+'.npz'))
    np.savez(args.output/'selected.npz', oof=cache['oof'], bagged_test=cache['test'],
             secondary_oof=secondary, train=train, full_test=np.clip(fitted.predict(x[n:]), 0, 5))
    summary = {'selected': best, 'secondary': metrics(y, secondary), 'training': metrics(y, train),
               'model':manifest['model'], 'revision':manifest['revision'], 'feature_name':args.feature_name,
               'fingerprint':fingerprint,
               'limitation':'Development selection and secondary splits reuse labels; neither is an untouched holdout.'}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
