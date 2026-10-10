"""Cache exact chunk boundaries and layer-10 inputs for fold-safe fine-tuning.

Layers 1-10 remain frozen. The cached inputs allow layers 11-12 to be trained
later using training-fold labels only, without repeatedly running the prefix.
"""
import argparse
import hashlib
import json
import os
import shutil
import time
from math import gcd
from pathlib import Path
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import WavLMModel
from extract_wavlm_frames import MODEL, REVISION


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--audio-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model=WavLMModel.from_pretrained(MODEL,revision=REVISION).to(device).eval()
    probe=torch.randn(1,16000,device=device)
    with torch.inference_mode():
        expected=model(probe,output_hidden_states=True).hidden_states[10]
    model.encoder.layers=torch.nn.ModuleList(list(model.encoder.layers[:10]))
    captured={}
    handle=model.encoder.layers[-1].register_forward_hook(
        lambda module,inputs,output:captured.update(frames=output[0]))
    with torch.inference_mode():
        model(probe)
        torch.testing.assert_close(captured['frames'],expected)
    del probe,expected
    rows=[(split,path) for split in ['train','test'] for path in sorted((args.audio_root/split).glob('*.wav'))]
    assert rows
    if args.limit:
        rows=rows[:args.limit]
    records=[]
    started=time.monotonic()
    for i,(split,path) in enumerate(rows):
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        key=hashlib.sha256(f'{sha}|{MODEL}|{REVISION}|layer10|10s|fullframes-float16|v1'.encode()).hexdigest()
        dest=args.output/(key+'.npz')
        if not dest.exists():
            if shutil.disk_usage(args.output).free < 3*1024**3:
                raise OSError('Less than 3 GiB free; stopping safely with completed caches preserved.')
            audio,sr=sf.read(path,dtype='float32')
            if audio.ndim==2:
                audio=audio.mean(axis=1)
            if sr!=16000:
                audio=resample_poly(audio,16000//gcd(sr,16000),sr//gcd(sr,16000))
            pieces=[]
            for offset in range(0,len(audio),160000):
                chunk=audio[offset:offset+160000]
                if len(chunk)<400:
                    continue
                chunk=(chunk-chunk.mean())/np.sqrt(chunk.var()+1e-7)
                with torch.inference_mode():
                    model(torch.from_numpy(chunk).unsqueeze(0).to(device))
                    pieces.append(captured['frames'][0].float().cpu().numpy().astype('float16'))
            lengths=np.asarray([len(a) for a in pieces],dtype='int32')
            values=np.concatenate(pieces)
            assert np.isfinite(values).all() and values.shape[1]==1024
            with dest.with_suffix('.tmp').open('wb') as out:
                np.savez(out,hidden=values,lengths=lengths)
            dest.with_suffix('.tmp').replace(dest)
        with np.load(dest) as z:
            lengths=z['lengths']
        records.append(dict(split=split,filename=path.name,sha256=sha,cache_key=key,
                            frames=int(lengths.sum()),chunks=len(lengths)))
        if i==0 or (i+1)%10==0:
            print(f'{i+1}/{len(rows)} elapsed={time.monotonic()-started:.1f}s',flush=True)
    handle.remove()
    prefix='pilot' if args.limit else 'complete'
    (args.output/f'{prefix}_manifest.json').write_text(json.dumps(dict(
        model=MODEL,revision=REVISION,layer=10,hidden_size=1024,chunk_seconds=10,
        dtype='float16',probe_verified=True,records=records),indent=2))


if __name__=='__main__':
    main()
