"""Select a small, documented blend grid using development OOF predictions."""
from improve_models import *

def main():
 folds,y,sets,te=load()
 tab=pd.read_csv(OUT/'benchmark.csv').sort_values('rmse'); base=tab.iloc[0]['model']
 sb=pd.read_csv(OUT/'speech_benchmark.csv').sort_values('rmse'); speech=sb.iloc[0]['model']
 b=np.load(OUT/f'{base}.npz');s=np.load(OUT/f'{speech}.npz');rows=[]
 for w in [0,.25,.5,.75,1]:
  p=(1-w)*b['oof']+w*s['oof']
  rows.append({'speech_weight':w,**metrics(y,p),'fold_rmse':[metrics(y[folds.fold==f],p[folds.fold==f])['rmse'] for f in sorted(folds.fold.unique())]})
 best=min(rows,key=lambda r:r['rmse']);w=best['speech_weight'];p=(1-w)*b['oof']+w*s['oof'];t=(1-w)*b['test']+w*s['test']
 sub=pd.DataFrame({'filename':te.filename,'label':t});dataset=Path(os.environ.get('GRAMMAR_DATA_ROOT') or json.loads((A/'run_config.json').read_text())['dataset']);official=pd.read_csv(dataset/'test.csv')
 assert sub.filename.tolist()==official.filename.tolist() and len(sub)==216 and sub.filename.is_unique
 assert np.isfinite(t).all() and np.all((t>=0)&(t<=5))
 sub.to_csv(OUT/'Ronit_Mia_v2.csv',index=False)
 pd.DataFrame({'filename':folds.filename,'target':y,'fold':folds.fold,'prediction':p}).to_csv(OUT/'candidate_oof.csv',index=False)
 result={'base_model':base,'speech_model':speech,'selected':best,'blend_grid':rows,'original_oof_rmse':.660851610083379,'original_oof_pearson':.84577894179478,'selection_caveat':'Development folds reused for selection; not an unbiased nested estimate.','kaggle_score':None,'submission_rows':len(sub)}
 (OUT/'candidate_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
