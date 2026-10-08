"""Extract frozen WavLM features locally; no labels or remote audio service."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from math import gcd
from transformers import WavLMModel

REVISIONS = {
    'large': 'c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c',
    'base-plus': '4c66d4806a428f2e922ccfa1a962776e232d487b',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=REVISIONS, default='large')
    p.add_argument('--audio-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model_id = 'microsoft/wavlm-' + args.model
    revision = REVISIONS[args.model]
    layers = [6, 12, 18, 24] if args.model == 'large' else [3, 6, 9, 12]
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print('Loading', model_id, revision, device, flush=True)
    model = WavLMModel.from_pretrained(model_id, revision=revision).to(device).eval()
    rows = [(split, path) for split in ['train', 'test']
            for path in sorted((args.audio_root / split).glob('*.wav'))]
    assert rows, 'No WAV files under train/ and test/'
    if args.limit:
        rows = rows[:args.limit]
    vectors, manifest = [], []
    start = time.monotonic()
    for i, (split, source) in enumerate(rows):
        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        settings = f'{sha}|{model_id}|{revision}|{layers}|10s|meanstd|v1'
        key = hashlib.sha256(settings.encode()).hexdigest()
        dest = args.output / (key + '.npy')
        if dest.exists():
            vector = np.load(dest)
        else:
            a, sr = sf.read(source, dtype='float32')
            if a.ndim == 2:
                a = a.mean(axis=1)
            if sr != 16000:
                a = resample_poly(a, 16000 // gcd(sr, 16000), sr // gcd(sr, 16000))
            sums = squares = None
            count = 0
            for offset in range(0, len(a), 160000):
                chunk = a[offset:offset + 160000]
                if len(chunk) < 400:
                    continue
                chunk = (chunk - chunk.mean()) / np.sqrt(chunk.var() + 1e-7)
                with torch.inference_mode():
                    hidden = model(torch.from_numpy(chunk).unsqueeze(0).to(device),
                                   output_hidden_states=True).hidden_states
                    values = torch.cat([hidden[layer][0] for layer in layers], dim=-1).float().cpu().numpy()
                total = values.sum(axis=0, dtype=np.float64)
                total2 = np.square(values.astype(np.float64)).sum(axis=0)
                sums = total if sums is None else sums + total
                squares = total2 if squares is None else squares + total2
                count += len(values)
            assert count, f'No usable audio: {source.name}'
            mean = sums / count
            vector = np.r_[mean, np.sqrt(np.maximum(squares / count - mean ** 2, 0))].astype('float32')
            assert np.isfinite(vector).all()
            tmp = dest.with_suffix('.tmp')
            with tmp.open('wb') as f:
                np.save(f, vector)
            tmp.replace(dest)
        vectors.append(vector)
        manifest.append({'split': split, 'filename': source.name, 'sha256': sha, 'cache_key': key})
        if i == 0 or (i + 1) % 10 == 0:
            print(f'{i+1}/{len(rows)} elapsed={time.monotonic()-start:.1f}s', flush=True)
    prefix = 'pilot' if args.limit else 'complete'
    np.save(args.output / f'{prefix}_embeddings.npy', np.stack(vectors))
    (args.output / f'{prefix}_manifest.json').write_text(json.dumps({
        'model': model_id, 'revision': revision, 'layers': layers,
        'hidden_size': model.config.hidden_size, 'device': device,
        'pooling': 'frame-weighted mean/std', 'chunk_seconds': 10, 'records': manifest,
    }, indent=2))
    print('Completed', prefix, flush=True)


if __name__ == '__main__':
    main()
