"""Export aggregate results and an executable reproduction notebook, no private features."""
import argparse
import json
import shutil
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--blend',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    summary=json.loads((args.blend/'blend_summary.json').read_text())
    assert summary['passed_export_gate'], 'Do not publish an unvalidated candidate as the new model.'
    shutil.copyfile(args.blend/'blend_summary.json',args.output/'blend_summary.json')
    shutil.copyfile(args.blend/'Ronit_Mia_v3.csv',args.output/'Ronit_Mia_v3.csv')
    selected=summary['selected']
    report=f'''# Third-round speech scoring experiments

The selected candidate is `{selected['name']}`.

- Development OOF RMSE: **{selected['development']['rmse']:.6f}**; Pearson: **{selected['development']['pearson']:.6f}**.
- Secondary split RMSE: **{selected['secondary']['rmse']:.6f}**; Pearson: **{selected['secondary']['pearson']:.6f}**.
- Full-fit training RMSE: **{selected['training']['rmse']:.6f}**; Pearson: **{selected['training']['pearson']:.6f}** (in-sample).
- Previous ensemble development/secondary RMSE: **{summary['baseline']['development']['rmse']:.6f} / {summary['baseline']['secondary']['rmse']:.6f}**.

## Method and selection

We extracted frozen [WavLM large](https://huggingface.co/microsoft/wavlm-large) hidden layers 6, 12, 18 and 24 from all provided recordings. Ten-second chunks cover the full recording; frame-weighted means and standard deviations yield fixed features. Standardization and SVR/Ridge regressors are fitted separately inside each training fold. Audio-hash groups and the previous development folds are retained.

We also extracted frozen [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) encoder layers 8, 16, 24 and 32. All audio is processed in 30-second chunks; pooling excludes padded frames. The encoder ran in float16 on the local GPU.

We also tested frozen [GPT-2](https://huggingface.co/openai-community/gpt2) transcript embeddings and token-likelihood statistics. GPT-2 embeddings and mean-only Wav2Vec2 pooling did not beat the previous ensemble in the tested blend grid. Likelihood statistics improved the handcrafted CatBoost component. The final comparison uses a small fixed convex-blend grid, then checks the selected blend on the secondary split. Test predictions average full-data fitting and fold bagging for each new component.

Only competition recordings/transcripts and provided training labels are used. Pretrained models are pinned by revision. Audio and text inference run locally. No manual test labels or leaderboard-derived labels are used. The CSV follows all 216 official test IDs and contains finite predictions between 0 and 5.

## Limits

These are local development results, not Kaggle scores. The models and blends were selected using reused development folds, so selection optimism is possible. The second split is a stability check, not an untouched holdout. Speaker overlap may remain because speaker IDs are unavailable. No local score establishes a leaderboard rank. The last verified public score before this round was 0.4337.

## Reproduction

Complete the original notebook and improvement pipeline first, including `prepare_secondary_base.py` and `validate_candidate.py`. Run `stage_next_inputs.py` to assemble the private intermediate inputs. Extract WavLM and GPT-2 features, evaluate them, then run `blend_representations.py` and this report generator. The companion notebook contains the command sequence. Intermediate metadata, transcripts, embeddings and per-record training predictions must remain private.
'''
    kaggle=args.output/'kaggle_submission.json'
    if kaggle.exists():
        result=json.loads(kaggle.read_text())
        report+=f"\n## Verified Kaggle result\n\nPublic score: **{result['public_score']}**. Observed rank: **{result['observed_public_rank']}**, on {result['observed_on']}. Leading score at that observation: **{result['observed_leading_score']}**. Rank #1 has not been established unless explicitly verified.\n"
    else:
        report+='\n## Kaggle status\n\nThis candidate has not yet been verified on Kaggle.\n'
    (args.output/'report.md').write_text(report)
    source='''from pathlib import Path
import os, subprocess, sys
ROOT = Path.cwd()
# Complete the original and second-round notebooks first.
RUN_EXPERIMENTS = False
PRIVATE = ROOT / 'experiments/private_round3'
def run(script, *args):
    subprocess.run([sys.executable, str(ROOT/'experiments'/script), *map(str,args)], check=True)
if RUN_EXPERIMENTS:
    import json
    data_root = Path(json.loads((ROOT/'artifacts/run_config.json').read_text())['dataset'])
    # The audio root must contain train/ and test/ WAV directories.
    # GRAMMAR_AUDIO_ROOT can point to a local verified staging copy.
    audio_root = Path(os.environ.get('GRAMMAR_AUDIO_ROOT', str(data_root)))
    run('stage_next_inputs.py', '--output', PRIVATE/'metadata')
    run('extract_wavlm.py', '--model','large','--audio-root',audio_root,'--output',PRIVATE/'wavlm')
    run('evaluate_representation.py','--features',PRIVATE/'wavlm','--metadata',PRIVATE/'metadata','--output',PRIVATE/'shl-wavlm-results')
    run('extract_whisper_encoder.py','--audio-root',audio_root,'--output',PRIVATE/'whisper')
    run('evaluate_representation.py','--features',PRIVATE/'whisper','--metadata',PRIVATE/'metadata','--output',PRIVATE/'shl-whisper-results')
    run('extract_language_features.py','--metadata',PRIVATE/'metadata','--output',PRIVATE/'gpt2')
    run('evaluate_language_likelihood.py','--metadata',PRIVATE/'metadata','--language',PRIVATE/'gpt2','--output',PRIVATE/'shl-likelihood-results')
    run('blend_representations.py','--metadata',PRIVATE/'metadata','--candidate',PRIVATE/'shl-wavlm-results','--candidate',PRIVATE/'shl-likelihood-results','--candidate',PRIVATE/'shl-whisper-results','--output',PRIVATE/'blend')
    run('build_round3_report.py','--blend',PRIVATE/'blend','--output',ROOT/'experiments/results/round3')
'''
    cells=[{'cell_type':'markdown','id':'intro','metadata':{},'source':'# Third-round grammar scoring experiments\n\nThis notebook displays aggregate results generated from completed runs. Execution counts are unset. Set RUN_EXPERIMENTS=True to regenerate after completing the earlier pipelines.'},
           {'cell_type':'code','id':'run','metadata':{},'source':source,'execution_count':None,'outputs':[]},
           {'cell_type':'code','id':'report','metadata':{},'source':"from IPython.display import display, Markdown\ndisplay(Markdown((ROOT/'experiments/results/round3/report.md').read_text()))",'execution_count':None,
            'outputs':[{'output_type':'display_data','metadata':{},'data':{'text/markdown':report}}]}]
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}},'cells':cells}
    root=Path(__file__).resolve().parents[1]
    (root/'grammar_scoring_round3.ipynb').write_text(json.dumps(notebook,indent=1)+'\n')
    print(report,flush=True)


if __name__=='__main__':main()
