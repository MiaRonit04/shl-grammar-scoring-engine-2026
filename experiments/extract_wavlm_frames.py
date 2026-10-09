"""Cache label-independent WavLM layer-12 frames for supervised pooling.

The encoder is truncated after layer 12, and a hook captures its output before
the final encoder normalization, matching hidden_states[12] of the full model.
Adjacent groups of five 20-ms frames are averaged to bound disk/training cost.
"""
import argparse
import hashlib
import json
import os
import time
from math import gcd
from pathlib import Path

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import WavLMModel

MODEL = 'microsoft/wavlm-large'
REVISION = 'c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--audio-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = WavLMModel.from_pretrained(MODEL, revision=REVISION).to(device).eval()
    # Verify the truncated output against the original model on a fixed signal.
    probe = torch.randn(1, 16000, device=device)
    with torch.inference_mode():
        expected = model(probe, output_hidden_states=True).hidden_states[12]
    model.encoder.layers = torch.nn.ModuleList(list(model.encoder.layers[:12]))
    captured = {}
    hook = model.encoder.layers[-1].register_forward_hook(
        lambda module, inputs, output: captured.update(frames=output[0]))
    with torch.inference_mode():
        model(probe)
        torch.testing.assert_close(captured['frames'], expected)
    del expected, probe
    rows = [(split, path) for split in ['train', 'test']
            for path in sorted((args.audio_root/split).glob('*.wav'))]
    if args.limit:
        rows = rows[:args.limit]
    assert rows
    records = []
    started = time.monotonic()
    for i, (split, path) in enumerate(rows):
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        key = hashlib.sha256(f'{sha}|{MODEL}|{REVISION}|layer12|10s|avg5|v1'.encode()).hexdigest()
        dest = args.output/(key+'.npy')
        if not dest.exists():
            audio, sr = sf.read(path, dtype='float32')
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if sr != 16000:
                audio = resample_poly(audio, 16000//gcd(sr, 16000), sr//gcd(sr, 16000))
            pieces = []
            for offset in range(0, len(audio), 160000):
                chunk = audio[offset:offset+160000]
                if len(chunk) < 400:
                    continue
                chunk = (chunk-chunk.mean())/np.sqrt(chunk.var()+1e-7)
                with torch.inference_mode():
                    model(torch.from_numpy(chunk).unsqueeze(0).to(device))
                    frames = captured['frames'][0].float().cpu().numpy()
                # Include the last partial group instead of dropping audio.
                pooled = np.stack([frames[j:j+5].mean(axis=0) for j in range(0, len(frames), 5)])
                pieces.append(pooled)
            values = np.concatenate(pieces).astype('float16')
            assert values.ndim == 2 and values.shape[1] == 1024 and np.isfinite(values).all()
            with dest.with_suffix('.tmp').open('wb') as f:
                np.save(f, values)
            dest.with_suffix('.tmp').replace(dest)
        values = np.load(dest, mmap_mode='r')
        records.append(dict(split=split, filename=path.name, sha256=sha, cache_key=key, frames=len(values)))
        if i == 0 or (i+1) % 10 == 0:
            print(f'{i+1}/{len(rows)} elapsed={time.monotonic()-started:.1f}s', flush=True)
    hook.remove()
    prefix = 'pilot' if args.limit else 'complete'
    (args.output/f'{prefix}_manifest.json').write_text(json.dumps(dict(
        model=MODEL, revision=REVISION, hidden_size=1024, layer=12,
        chunk_seconds=10, frame_average=5, records=records,
        probe_verified=True, device=device), indent=2))


if __name__ == '__main__':
    main()
