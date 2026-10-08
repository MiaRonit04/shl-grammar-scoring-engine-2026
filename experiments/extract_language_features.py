"""Frozen local GPT-2 likelihood and hidden-state features of ASR transcripts."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = 'openai-community/gpt2'
REVISION = '607a30d783dfa663caf39e06633721c8d4cfcd7e'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, use_safetensors=True).eval()
    records, vectors, likelihood = [], [], []
    start = time.monotonic()
    for split in ['train', 'test']:
        frame = pd.read_csv(args.metadata/f'{split}_transcripts.csv').fillna({'transcript': ''})
        for row in frame.itertuples():
            text = row.transcript
            key = hashlib.sha256(f'{MODEL}|{REVISION}|512context64|v1|{text}'.encode()).hexdigest()
            dest = args.output/(key+'.npz')
            if dest.exists():
                cache = np.load(dest)
                embedding, stats = cache['embedding'], cache['stats']
            else:
                ids = tokenizer.encode(text, add_special_tokens=False)
                values, losses = [], []
                for offset in range(0, len(ids), 448):
                    begin = max(0, offset-64)
                    segment = ids[begin:offset+448]
                    if len(segment) < 2:
                        continue
                    tokens = torch.tensor([segment])
                    with torch.inference_mode():
                        result = model(tokens, output_hidden_states=True, use_cache=False)
                        nll = torch.nn.functional.cross_entropy(result.logits[0, :-1].float(), tokens[0, 1:], reduction='none')
                        fresh = max(1, offset-begin)
                        losses.append(nll[fresh-1:].numpy())
                        values.append(torch.cat([result.hidden_states[k][0, fresh:] for k in [6, 12]], dim=-1).numpy())
                if not losses:
                    embedding = np.zeros(3072, dtype='float32')
                    stats = np.r_[np.zeros(13), 1].astype('float32')
                else:
                    losses = np.concatenate(losses)
                    values = np.concatenate(values)
                    embedding = np.r_[values.mean(0), values.std(0)].astype('float32')
                    stats = np.r_[losses.mean(), losses.std(), np.quantile(losses, [.1,.25,.5,.75,.9,.95]),
                                  [(losses > threshold).mean() for threshold in [3,5,7,10]], np.log1p(len(losses)), 0].astype('float32')
                assert embedding.shape == (3072,) and stats.shape == (14,)
                assert np.isfinite(embedding).all() and np.isfinite(stats).all()
                np.savez(dest, embedding=embedding, stats=stats)
            records.append({'split': split, 'filename': row.filename, 'sha256': row.audio_hash, 'text_cache_key': key})
            vectors.append(embedding)
            likelihood.append(stats)
            if len(records) % 20 == 0:
                print(f'{len(records)}/985 elapsed={time.monotonic()-start:.1f}s', flush=True)
    np.save(args.output/'complete_embeddings.npy', np.stack(vectors))
    np.save(args.output/'likelihood.npy', np.stack(likelihood))
    (args.output/'complete_manifest.json').write_text(json.dumps({
        'model': MODEL, 'revision': REVISION, 'layers': [6,12], 'hidden_size': 768,
        'records': records, 'device': 'cpu', 'pooling': 'token mean/std',
        'likelihood_columns': ['nll_mean','nll_std','nll_p10','nll_p25','nll_p50','nll_p75','nll_p90','nll_p95',
                               'fraction_over3','fraction_over5','fraction_over7','fraction_over10','log_token_count','empty'],
    }, indent=2))
    print('Completed language features', flush=True)


if __name__ == '__main__':
    main()
