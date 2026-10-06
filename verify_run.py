"""Validate computed notebook artifacts against source IDs, labels and saved OOF arrays."""
import json,math
from pathlib import Path
import numpy as np
import pandas as pd
import nbformat
from scipy.stats import pearsonr

out=Path('artifacts')
nb=nbformat.read('grammar_scoring_engine.ipynb',as_version=4)
nbformat.validate(nb)
code=[c for c in nb.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in code),'Some code cells were not executed'
assert not [(i,o) for i,c in enumerate(code) for o in c.outputs if o.output_type=='error'],'Notebook contains execution errors'
config=json.loads((out/'run_config.json').read_text())
root=Path(config['dataset'])
train=pd.read_csv(root/'train.csv'); test=pd.read_csv(root/'test.csv'); template=pd.read_csv(root/'sample_submission.csv')
folds=pd.read_csv(out/'folds.csv')
oof=pd.read_csv(out/'oof_predictions.csv')
results=pd.read_csv(out/'model_results.csv')
metrics=json.loads((out/'final_metrics.json').read_text())
pred=pd.read_csv(out/'test_predictions.csv')
assert oof.filename.tolist()==train.filename.tolist()
assert pred.iloc[:,0].tolist()==test.filename.tolist()
assert np.allclose(oof.target,train.label)
assert folds.filename.tolist()==train.filename.tolist()
assert folds.groupby('audio_hash').fold.nunique().max()==1
for _,r in results.iterrows():
    p=oof[r.Model].to_numpy(); assert np.isfinite(p).all()
    rmse=np.sqrt(np.mean((oof.target-p)**2))
    assert np.isclose(rmse,r.RMSE)
    if np.std(p)>1e-12: assert np.isclose(pearsonr(oof.target,p).statistic,r.Pearson,equal_nan=True)
training_predictions=pd.read_csv(out/'training_predictions.csv')
assert training_predictions.filename.tolist()==train.filename.tolist()
assert np.isclose(np.sqrt(np.mean((training_predictions.target-training_predictions.prediction)**2)),metrics['training']['RMSE'])
assert np.isclose(pearsonr(training_predictions.target,training_predictions.prediction).statistic,metrics['training']['Pearson'])
selected=results.loc[results.Model==metrics['model']].iloc[0]
assert np.isclose(selected.RMSE,metrics['oof']['RMSE'])
assert np.isfinite(pred.iloc[:,1]).all() and pred.iloc[:,1].between(0,5).all()
for split,frame in [('train',train),('test',test)]:
    transcripts=pd.read_csv(out/f'{split}_transcripts.csv',keep_default_na=False)
    assert transcripts.filename.tolist()==frame.filename.tolist()
    assert transcripts.asr_error.ne('').mean()<=.05
emb=np.load(out/'text_embeddings.npy')
assert emb.shape[0]==len(train)+len(test) and np.isfinite(emb).all()
export_record=out/'submission_export.json'
submission_file=out/'submission.csv'
if export_record.exists():
    export=json.loads(export_record.read_text())
    submission_file=Path(export['path'])
    template=pd.read_csv(export['template'])
    config['submission_written']=True
    config['template_compatible']=export['template_compatible']
    assert export['template_compatible'] or export['explicit_test_order']
if config['submission_written']:
    submission=pd.read_csv(submission_file)
    assert len(submission)==len(test) and submission.columns.tolist()==template.columns.tolist()
    expected=template.iloc[:,0].tolist() if config['template_compatible'] else test.iloc[:,0].tolist()
    assert submission.iloc[:,0].tolist()==expected
    mapped=pred.set_index(pred.columns[0]).iloc[:,0].reindex(expected).to_numpy()
    assert np.allclose(submission.iloc[:,1],mapped)
else:
    assert not config['template_compatible'] and config['SUBMISSION_POLICY']=='strict'
    assert not (out/'submission.csv').exists(),'Stale submission.csv must not be mistaken for this run'
summary={'notebook_code_cells_executed':len(code),'experiments_verified':len(results),'train_rows':len(train),'test_rows':len(test),
         'selected_model':metrics['model'],'training_metrics':metrics['training'],'oof_metrics':metrics['oof'],
         'submission_written':config['submission_written'],'all_checks_passed':True}
(out/'verification_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
