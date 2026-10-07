"""Fold-safe evaluation of frozen speech representations and multimodal regressors."""
from improve_models import *
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

def balance_blocks(x, split):
    return np.c_[x[:, :split] / np.sqrt(split), x[:, split:] / np.sqrt(x.shape[1]-split)]

def speech_specs(speech, basic):
    specs={}
    for layer,k in [('6',0),('9',1),('12',2)]:
        x=np.c_[speech[:,k*768:(k+1)*768],speech[:,2304+k*768:2304+(k+1)*768]]
        for c in [1,3,10]:
            pipe=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),SVR(C=c,epsilon=.1))
            specs[f'speech{layer}_svr{c}']=(x,pipe)
        for alpha in [100,1000,10000]:
            pipe=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=alpha))
            specs[f'speech{layer}_ridge{alpha}']=(x,pipe)
        joint=np.c_[x,basic];split=x.shape[1]
        for c in [3,10]:
            pipe=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),FunctionTransformer(balance_blocks,kw_args={'split':split}),SVR(C=c,epsilon=.1))
            specs[f'speech{layer}_joint_svr{c}']=(joint,pipe)
        prep=ColumnTransformer([
            ('speech',make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler(),PCA(n_components=32,svd_solver='full')),list(range(split))),
            ('basic',SimpleImputer(strategy='median',keep_empty_features=True),list(range(split,joint.shape[1])))])
        model=CatBoostRegressor(iterations=1000,depth=4,learning_rate=.035,l2_leaf_reg=10,loss_function='RMSE',random_seed=42,verbose=False,thread_count=4,allow_writing_files=False)
        specs[f'speech{layer}_joint_cat']=(joint,make_pipeline(prep,model))
    return specs

def main():
    folds,y,sets,te=load();n=len(y);speech=np.load(ROOT/'experiments/speech_embeddings.npy')
    manifest=json.loads((ROOT/'experiments/speech_manifest.json').read_text())
    assert manifest['filenames']==folds.filename.tolist()+te.filename.tolist()
    results=[]
    for name,(x,pipe) in speech_specs(speech,sets['basic']).items():
        dest=OUT/f'{name}.npz'
        if dest.exists():p=np.load(dest)['oof']
        else:
            p=np.zeros(n);tp=[]
            for f in sorted(folds.fold.unique()):
                tr=np.where(folds.fold!=f)[0];va=np.where(folds.fold==f)[0]
                model=clone(pipe).fit(x[tr],y[tr]);p[va]=np.clip(model.predict(x[va]),0,5);tp.append(np.clip(model.predict(x[n:]),0,5))
            np.savez(dest,oof=p,test=np.mean(tp,axis=0))
        row={'model':name,**metrics(y,p),'fold_rmse':[metrics(y[folds.fold==f],p[folds.fold==f])['rmse'] for f in sorted(folds.fold.unique())]}
        results.append(row);print(json.dumps(row),flush=True)
        pd.DataFrame(results).sort_values('rmse').to_csv(OUT/'speech_benchmark.csv',index=False)
if __name__=='__main__':main()
