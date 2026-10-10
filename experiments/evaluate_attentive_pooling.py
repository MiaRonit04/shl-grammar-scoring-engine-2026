"""Fold-safe supervised attentive pooling over frozen WavLM frames.

Uses only competition training labels. Two fixed training durations (20/50
epochs), each averaged over seeds 42/2026. Secondary splits are evaluated only
after development selection. Inspired by https://arxiv.org/abs/1803.10963;
this is a small regression adaptation, not a reproduction of its experiments.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from evaluate_representation import metrics

MAX_FRAMES = 256
BATCH_SIZE = 32
EPOCHS = (20, 50)
SEEDS = (42, 2026)


class Scorer(torch.nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.project = torch.nn.Sequential(torch.nn.Linear(dim, 64), torch.nn.GELU())
        self.attention = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
        self.head = torch.nn.Sequential(torch.nn.Dropout(.25), torch.nn.Linear(128, 64),
                                        torch.nn.GELU(), torch.nn.Dropout(.25), torch.nn.Linear(64, 1))

    def forward(self, x, mask):
        values = self.project(x)
        logits = self.attention(values).squeeze(-1).masked_fill(~mask, -1e4)
        weights = logits.softmax(dim=1).unsqueeze(-1)
        mean = (weights*values).sum(dim=1)
        var = (weights*values.square()).sum(dim=1)-mean.square()
        pooled = torch.cat([mean, var.clamp_min(1e-5).sqrt()], dim=-1)
        return self.head(pooled).squeeze(-1)


def load_frames(folder, metadata):
    manifest = json.loads((folder/'complete_manifest.json').read_text())
    folds = pd.read_csv(metadata/'folds.csv')
    test = pd.read_csv(metadata/'test_transcripts.csv')
    audit = pd.read_csv(metadata/'audio_audit.csv')
    hashes = {(r.split, r.filename): r.sha256 for r in audit.itertuples()}
    records = {(r['split'], r['filename']): r for r in manifest['records']}
    order = [('train', name) for name in folds.filename] + [('test', name) for name in test.filename]
    assert len(records) == len(order) == len(manifest['records'])
    x = np.zeros((len(order), MAX_FRAMES, 1024), dtype='float16')
    lengths = np.empty(len(order), dtype='int64')
    # Moments are sample-balanced, then aggregated using training rows only.
    means, squares = [], []
    for i, key in enumerate(order):
        r = records[key]
        assert hashes[key] == r['sha256']
        a = np.load(folder/(r['cache_key']+'.npy')).astype('float32')
        assert a.ndim == 2 and a.shape[1] == 1024 and len(a) > 0 and np.isfinite(a).all()
        if len(a) > MAX_FRAMES:
            a = np.stack([part.mean(axis=0) for part in np.array_split(a, MAX_FRAMES)])
        lengths[i] = len(a)
        x[i, :len(a)] = a
        means.append(a.mean(axis=0))
        squares.append(np.square(a).mean(axis=0))
    return x, lengths, np.array(means), np.array(squares), folds, manifest


def fit_run(x, lengths, means, squares, y, tr, predict_ids, seed, epochs, device, save_path=None):
    """No validation targets are visible to the optimizer or normalization."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean = means[tr].mean(axis=0)
    std = np.sqrt(np.maximum(squares[tr].mean(axis=0)-mean**2, 1e-4))
    center, scale = float(y[tr].mean()), float(y[tr].std())
    assert scale > 0
    mean_t, std_t = [torch.from_numpy(a).to(device) for a in (mean, std)]
    target = torch.from_numpy(((y-center)/scale).astype('float32')).to(device)
    model = Scorer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=3e-5)

    def batch(ids):
        size = int(lengths[ids].max())
        a = torch.from_numpy(x[ids, :size].astype('float32')).to(device)
        mask = torch.arange(size, device=device)[None, :] < torch.as_tensor(lengths[ids], device=device)[:, None]
        return (a-mean_t)/std_t, mask

    output = {}
    for epoch in range(1, max(epochs)+1):
        model.train()
        shuffled = rng.permutation(tr)
        for offset in range(0, len(tr), BATCH_SIZE):
            ids = shuffled[offset:offset+BATCH_SIZE]
            a, mask = batch(ids)
            opt.zero_grad(set_to_none=True)
            pred = model(a, mask)
            loss = torch.nn.functional.mse_loss(pred, target[torch.as_tensor(ids, device=device)])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()
        if epoch in epochs:
            model.eval()
            preds = []
            with torch.inference_mode():
                for offset in range(0, len(predict_ids), BATCH_SIZE):
                    ids = predict_ids[offset:offset+BATCH_SIZE]
                    a, mask = batch(ids)
                    preds.extend((model(a, mask)*scale+center).cpu().numpy().tolist())
            output[epoch] = np.clip(np.asarray(preds), 0, 5)
            assert np.isfinite(output[epoch]).all()
        if epoch % 10 == 0:
            print(f'  seed={seed} epoch={epoch}/{max(epochs)}', flush=True)
    if save_path:
        torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},
                        mean=torch.from_numpy(mean), std=torch.from_numpy(std),
                        target_center=center, target_scale=scale,
                        epochs=max(epochs), seed=seed, max_frames=MAX_FRAMES), save_path)
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    x, lengths, means, squares, folds, manifest = load_frames(args.features, args.metadata)
    y = folds.target.to_numpy()
    n = len(y)
    groups = folds.audio_hash.to_numpy()
    split = [(np.flatnonzero(folds.fold != f), np.flatnonzero(folds.fold == f)) for f in sorted(folds.fold.unique())]
    digest = hashlib.sha256(Path(__file__).read_bytes()+json.dumps(manifest, sort_keys=True).encode()
                            +y.tobytes()+folds.fold.to_numpy().tobytes()).hexdigest()
    oof = {e:np.zeros(n) for e in EPOCHS}
    bagged = {e:[] for e in EPOCHS}
    start = time.monotonic()

    def cached_run(name, tr, ids, seed, epochs, save_path=None):
        dest = args.output/(name+'.npz')
        if dest.exists():
            z = np.load(dest)
            assert str(z['fingerprint']) == digest
            return {e:z[str(e)] for e in epochs}
        save_path = save_path or args.output/(name+'.pt')
        values = fit_run(x, lengths, means, squares, y, tr, ids, seed, epochs, device, save_path)
        np.savez(dest, fingerprint=digest, **{str(e):a for e,a in values.items()})
        return values

    for fold, (tr, va) in enumerate(split):
        assert not set(groups[tr]) & set(groups[va])
        ids = np.r_[va, np.arange(n, len(x))]
        for seed in SEEDS:
            values = cached_run(f'development_fold{fold}_seed{seed}', tr, ids, seed, EPOCHS)
            for e, pred in values.items():
                oof[e][va] += pred[:len(va)]/len(SEEDS)
                bagged[e].append(pred[len(va):])
        print(f'Completed development fold {fold} elapsed={time.monotonic()-start:.1f}s', flush=True)
    results = [dict(model=f'attentive_epochs{e}_two_seeds', epochs=e, **metrics(y, oof[e]),
                    folds=[metrics(y[va], oof[e][va]) for _, va in split]) for e in EPOCHS]
    (args.output/'benchmark.json').write_text(json.dumps(results, indent=2))
    best = min(results, key=lambda r:r['rmse'])
    epoch = best['epochs']
    secondary = np.zeros(n)
    bins = pd.qcut(y, 5, labels=False, duplicates='drop')
    for fold, (tr, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=2026).split(x[:n], bins, groups)):
        assert not set(groups[tr]) & set(groups[va])
        for seed in SEEDS:
            values = cached_run(f'secondary_fold{fold}_epoch{epoch}_seed{seed}', tr, va, seed, (epoch,))
            secondary[va] += values[epoch]/len(SEEDS)
        print(f'Completed secondary fold {fold} elapsed={time.monotonic()-start:.1f}s', flush=True)
    full = []
    for seed in SEEDS:
        values = cached_run(f'full_epoch{epoch}_seed{seed}', np.arange(n), np.arange(len(x)), seed, (epoch,),
                            args.output/f'model_seed{seed}.pt')
        full.append(values[epoch])
    full = np.mean(full, axis=0)
    np.savez(args.output/'selected.npz', oof=oof[epoch], secondary_oof=secondary,
             train=full[:n], full_test=full[n:], bagged_test=np.mean(bagged[epoch], axis=0))
    summary = dict(selected=best, secondary=metrics(y, secondary), training=metrics(y, full[:n]),
                   model=manifest['model'], revision=manifest['revision'], fingerprint=digest,
                   method='frozen layer12 frames, supervised attentive mean/std pooling', device=device,
                   normalization='fit only on fold training rows', seeds=SEEDS,
                   limitation='Repeated development selection; secondary splits reuse labels and are not an untouched holdout.')
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
