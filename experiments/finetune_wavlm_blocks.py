"""Fine-tune WavLM layers 11-12 from label-independent cached layer-10 inputs.

Every outer fold starts from the same pinned speech weights and a scoring head
trained only on that fold's training rows. No test targets or external dataset.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from transformers import WavLMModel
from extract_wavlm_frames import MODEL, REVISION
from evaluate_attentive_pooling import Scorer, load_frames, fit_run, MAX_FRAMES
from evaluate_representation import metrics

EPOCHS=(2,4)


def reduce_frames(frames, limit=MAX_FRAMES):
    """Differentiable equivalent of numpy.array_split followed by means."""
    if len(frames)<=limit:
        return frames
    sizes=np.full(limit,len(frames)//limit,dtype='int64')
    sizes[:len(frames)%limit]+=1
    groups=torch.as_tensor(np.repeat(np.arange(limit),sizes),device=frames.device)
    result=torch.zeros(limit,frames.shape[-1],device=frames.device,dtype=frames.dtype)
    result=result.index_add(0,groups,frames)
    return result/torch.as_tensor(sizes,device=frames.device)[:,None]


class PartialSpeechScorer(torch.nn.Module):
    def __init__(self, warmup, device):
        super().__init__()
        base=WavLMModel.from_pretrained(MODEL,revision=REVISION)
        self.layers=torch.nn.ModuleList(list(base.encoder.layers[10:12]))
        self.bias_attention=base.encoder.layers[0].attention.requires_grad_(False)
        self.head=Scorer()
        self.head.load_state_dict(warmup['state_dict'])
        self.register_buffer('mean',warmup['mean'])
        self.register_buffer('std',warmup['std'])
        self.center=float(warmup['target_center'])
        self.scale=float(warmup['target_scale'])
        self.bias_cache={}
        del base
        self.to(device)

    def frame_features(self, path):
        with np.load(path) as cache:
            values=cache['hidden'].astype('float32')
            lengths=cache['lengths']
        pieces=[]
        offset=0
        for length in lengths:
            length=int(length)
            if length not in self.bias_cache:
                # A cache populated during inference must remain usable by autograd.
                with torch.inference_mode(False),torch.no_grad():
                    self.bias_cache[length]=self.bias_attention.compute_bias(length,length).detach()
            hidden=torch.from_numpy(values[offset:offset+length]).unsqueeze(0).to(self.mean.device)
            offset+=length
            bias=self.bias_cache[length]
            for layer in self.layers:
                hidden,bias=layer(hidden,position_bias=bias)
            pooled=torch.nn.functional.avg_pool1d(hidden.transpose(1,2),5,stride=5,
                                                  ceil_mode=True,count_include_pad=False)
            pieces.append(pooled[0].transpose(0,1))
        assert offset==len(values)
        return reduce_frames(torch.cat(pieces,dim=0))

    def forward(self,path):
        frames=(self.frame_features(path)-self.mean)/self.std
        mask=torch.ones(1,len(frames),device=frames.device,dtype=torch.bool)
        return self.head(frames.unsqueeze(0),mask)[0]


def fit_blocks(paths,y,tr,ids,warmup,epochs,seed,device,checkpoint):
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    model=PartialSpeechScorer(warmup,device)
    opt=torch.optim.AdamW([{'params':model.layers.parameters(),'lr':1e-5},
                          {'params':model.head.parameters(),'lr':3e-4}],weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=4,eta_min=1e-6)
    trainable=[p for p in model.parameters() if p.requires_grad]
    output={}
    for epoch in range(1,max(epochs)+1):
        model.train()
        perm=rng.permutation(tr)
        losses=[]
        for start in range(0,len(perm),8):
            batch=perm[start:start+8]
            opt.zero_grad(set_to_none=True)
            for i in batch:
                prediction=model(paths[i])
                target=(float(y[i])-model.center)/model.scale
                loss=(prediction-target).square()
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite encoder fine-tuning loss')
                (loss/len(batch)).backward()
                losses.append(float(loss.detach().cpu()))
            torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
            opt.step()
        scheduler.step()
        print(f'fine-tune epoch={epoch} standardized training MSE={np.mean(losses):.6f}',flush=True)
        if epoch in epochs:
            model.eval()
            with torch.inference_mode():
                pred=[float(model(paths[i]).cpu())*model.scale+model.center for i in ids]
            output[epoch]=np.clip(np.asarray(pred),0,5)
            assert np.isfinite(output[epoch]).all()
    torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},
                    center=model.center,scale=model.scale,model=MODEL,revision=REVISION,
                    epochs=max(epochs),seed=seed),checkpoint)
    del model,opt
    gc.collect()
    if device=='mps':
        torch.mps.empty_cache()
    elif device=='cuda':
        torch.cuda.empty_cache()
    return output


def smoke(args,device):
    manifest=json.loads((args.blocks/'pilot_manifest.json').read_text())
    record=manifest['records'][0]
    path=args.blocks/(record['cache_key']+'.npz')
    warmup=torch.load(args.warmup/'development_fold0_seed42.pt',map_location='cpu',weights_only=True)
    torch.manual_seed(42)
    model=PartialSpeechScorer(warmup,device).eval()
    # Compare the frozen block reconstruction with the original layer-12 cache.
    frames_manifest=json.loads((args.features/'complete_manifest.json').read_text())
    source=next(r for r in frames_manifest['records']
                if r['split']==record['split'] and r['filename']==record['filename'])
    expected=np.load(args.features/(source['cache_key']+'.npy')).astype('float32')
    if len(expected)>MAX_FRAMES:
        expected=np.stack([a.mean(0) for a in np.array_split(expected,MAX_FRAMES)])
    with torch.inference_mode():
        actual=model.frame_features(path)
        torch.testing.assert_close(actual.cpu(),torch.from_numpy(expected),atol=.02,rtol=.01)
        delta=float((actual.cpu()-torch.from_numpy(expected)).abs().max())
    model.train()
    before=model.layers[0].attention.q_proj.weight.detach().clone()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5)
    loss=(model(path)-.25).square()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(p.grad is None for p in model.bias_attention.parameters())
    opt.step()
    assert not torch.equal(before,model.layers[0].attention.q_proj.weight)
    print(json.dumps(dict(smoke_passed=True,reconstruction_max_absolute_error=delta,
                          finite_gradients=True,frozen_bias_unchanged=True)),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--blocks',type=Path,required=True)
    parser.add_argument('--features',type=Path,required=True)
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--warmup',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke-test',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if args.smoke_test:
        smoke(args,device)
        return
    manifest=json.loads((args.blocks/'complete_manifest.json').read_text())
    previous=json.loads((args.warmup/'summary.json').read_text())
    warmup_epochs=previous['selected']['epochs']
    x,lengths,means,squares,folds,frame_manifest=load_frames(args.features,args.metadata)
    test=pd.read_csv(args.metadata/'test_transcripts.csv')
    y=folds.target.to_numpy()
    n=len(y)
    records={(r['split'],r['filename']):r for r in manifest['records']}
    expected={(r['split'],r['filename']):r['sha256'] for r in frame_manifest['records']}
    order=[('train',name) for name in folds.filename]+[('test',name) for name in test.filename]
    assert len(records)==len(manifest['records'])==len(order)
    assert all(records[key]['sha256']==expected[key] for key in order)
    paths=[args.blocks/(records[key]['cache_key']+'.npz') for key in order]
    digest=hashlib.sha256(Path(__file__).read_bytes()+json.dumps(manifest,sort_keys=True).encode()
                          +previous['fingerprint'].encode()+y.tobytes()+folds.fold.to_numpy().tobytes()).hexdigest()
    groups=folds.audio_hash.to_numpy()

    def run(name,old_name,tr,ids,epochs):
        dest=args.output/(name+'.npz')
        if dest.exists():
            with np.load(dest) as z:
                assert str(z['fingerprint'])==digest
                return {e:z[str(e)] for e in epochs}
        source=args.warmup/(old_name+'.pt')
        warmup=torch.load(source,map_location='cpu',weights_only=True)
        if warmup['epochs']!=warmup_epochs:
            source=args.output/(name+'_warmup.pt')
            fit_run(x,lengths,means,squares,y,tr,np.array([],dtype='int64'),42,(warmup_epochs,),device,source)
            warmup=torch.load(source,map_location='cpu',weights_only=True)
        assert np.isclose(warmup['target_center'],y[tr].mean(),atol=1e-8)
        assert np.isclose(warmup['target_scale'],y[tr].std(),atol=1e-8)
        value=fit_blocks(paths,y,tr,ids,warmup,epochs,42,device,args.output/(name+'.pt'))
        np.savez(dest,fingerprint=digest,**{str(e):a for e,a in value.items()})
        return value

    splits=[(np.flatnonzero(folds.fold!=f),np.flatnonzero(folds.fold==f)) for f in sorted(folds.fold.unique())]
    oof={e:np.zeros(n) for e in EPOCHS}
    bagged={e:[] for e in EPOCHS}
    for fold,(tr,va) in enumerate(splits):
        assert not set(groups[tr]) & set(groups[va])
        ids=np.r_[va,np.arange(n,len(paths))]
        values=run(f'development_fold{fold}',f'development_fold{fold}_seed42',tr,ids,EPOCHS)
        for e,pred in values.items():
            oof[e][va]=pred[:len(va)]
            bagged[e].append(pred[len(va):])
        print(f'Completed fine-tuning development fold {fold}',flush=True)
    results=[dict(model=f'finetune_two_blocks_epoch{e}',epochs=e,**metrics(y,oof[e]),
                  folds=[metrics(y[va],oof[e][va]) for _,va in splits]) for e in EPOCHS]
    (args.output/'benchmark.json').write_text(json.dumps(results,indent=2))
    best=min(results,key=lambda r:r['rmse'])
    epoch=best['epochs']
    secondary=np.zeros(n)
    bins=pd.qcut(y,5,labels=False,duplicates='drop')
    for fold,(tr,va) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=2026).split(x[:n],bins,groups)):
        assert not set(groups[tr]) & set(groups[va])
        name=f'secondary_fold{fold}'
        old=f'secondary_fold{fold}_epoch{warmup_epochs}_seed42'
        secondary[va]=run(name,old,tr,va,(epoch,))[epoch]
    full=run('full', 'model_seed42',np.arange(n),np.arange(len(paths)),(epoch,))[epoch]
    np.savez(args.output/'selected.npz',oof=oof[epoch],secondary_oof=secondary,train=full[:n],
             full_test=full[n:],bagged_test=np.mean(bagged[epoch],axis=0))
    summary=dict(selected=best,secondary=metrics(y,secondary),training=metrics(y,full[:n]),
                 model=MODEL,revision=REVISION,fingerprint=digest,method='WavLM layers11-12 fine-tuning',
                 warmup_epochs=warmup_epochs,seed=42,
                 limitation='Reused development labels; secondary split is a stability check, not an untouched holdout.')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
