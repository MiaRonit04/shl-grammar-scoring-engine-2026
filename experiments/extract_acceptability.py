"""Local frozen CoLA-model features from existing ASR transcripts.

Model: https://huggingface.co/textattack/roberta-base-CoLA
No CoLA dataset is downloaded. A permitted pretrained model supplies features,
not competition labels. Class probabilities are not calibrated grammar scores.
"""
import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = 'textattack/roberta-base-CoLA'
REVISION = '3ccf3a400f2fa75ff257eac171047603ffbe84f1'
COLUMNS = ['prob_mean','prob_std','prob_p10','prob_p25','prob_p50','prob_p75','prob_p90',
           'fraction_under_025','fraction_under_05','token_weighted_probability',
           'log_segment_count','log_token_count','empty']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REVISION).eval()
    records, embeddings, statistics = [], [], []
    start = time.monotonic()
    for split in ['train','test']:
        table = pd.read_csv(args.metadata/f'{split}_transcripts.csv').fillna({'transcript':''})
        for row in table.itertuples():
            key = hashlib.sha256(f'{MODEL}|{REVISION}|sentences126|v1|{row.transcript}'.encode()).hexdigest()
            dest = args.output/(key+'.npz')
            if dest.exists():
                z = np.load(dest)
                vector, stats = z['embedding'], z['stats']
            else:
                segments = []
                for sentence in re.split(r'(?<=[.!?])\s+', row.transcript.strip()):
                    tokens = tokenizer.encode(sentence, add_special_tokens=False)
                    for offset in range(0,len(tokens),126):
                        segments.append(tokens[offset:offset+126])
                if not segments:
                    vector = np.zeros(1536, dtype='float32')
                    stats = np.r_[np.zeros(12),1].astype('float32')
                else:
                    lengths = np.asarray([len(a) for a in segments])
                    probabilities, hidden = [], []
                    for offset in range(0,len(segments),8):
                        batch = segments[offset:offset+8]
                        prepared = [{'input_ids':[tokenizer.cls_token_id,*a,tokenizer.sep_token_id]} for a in batch]
                        tokens = tokenizer.pad(prepared,padding=True,return_tensors='pt')
                        with torch.inference_mode():
                            result = model(**tokens,output_hidden_states=True)
                            probabilities.extend(result.logits.softmax(-1)[:,1].numpy().tolist())
                            hidden.extend(result.hidden_states[-1][:,0].numpy())
                    probabilities = np.asarray(probabilities)
                    hidden = np.asarray(hidden)
                    mean = np.average(hidden,axis=0,weights=lengths)
                    std = np.sqrt(np.maximum(np.average((hidden-mean)**2,axis=0,weights=lengths),0))
                    vector = np.r_[mean,std].astype('float32')
                    stats = np.r_[probabilities.mean(),probabilities.std(),np.quantile(probabilities,[.1,.25,.5,.75,.9]),
                                  (probabilities<.25).mean(),(probabilities<.5).mean(),
                                  np.average(probabilities,weights=lengths),np.log1p(len(segments)),
                                  np.log1p(lengths.sum()),0].astype('float32')
                assert vector.shape==(1536,) and stats.shape==(len(COLUMNS),)
                assert np.isfinite(vector).all() and np.isfinite(stats).all()
                np.savez(dest,embedding=vector,stats=stats)
            records.append(dict(split=split,filename=row.filename,sha256=row.audio_hash,text_cache_key=key))
            embeddings.append(vector)
            statistics.append(stats)
            if len(records)%20==0:
                print(f'{len(records)}/985 elapsed={time.monotonic()-start:.1f}s',flush=True)
    np.save(args.output/'complete_embeddings.npy',np.stack(embeddings))
    np.save(args.output/'acceptability.npy',np.stack(statistics))
    (args.output/'complete_manifest.json').write_text(json.dumps(dict(
        model=MODEL,revision=REVISION,layers=[12],hidden_size=768,records=records,device='cpu',
        pooling='token-weighted sentence CLS mean/std',feature_columns=COLUMNS,
        limitation='Written-sentence acceptability and ASR text may not reflect spoken grammar faithfully.'),indent=2))


if __name__=='__main__':
    main()
