"""Cache the fixed secondary-split baseline comparison while speech is extracted."""
from validate_candidate import *

def main():
    folds,y,sets,te=load();n=len(y)
    name=pd.read_csv(OUT/'benchmark.csv').sort_values('rmse').iloc[0]['model']
    feature,base=base_estimator(name)
    old=CatBoostRegressor(iterations=400,depth=4,learning_rate=.035,l2_leaf_reg=10,loss_function='RMSE',random_seed=42,verbose=False,thread_count=4,allow_writing_files=False)
    hist=HistGradientBoostingRegressor(max_iter=150,learning_rate=.04,max_leaf_nodes=10,l2_regularization=8,early_stopping=False,random_state=42)
    bins=pd.qcut(y,5,labels=False,duplicates='drop');splits=list(StratifiedGroupKFold(5,shuffle=True,random_state=2026).split(np.zeros(n),bins,folds.audio_hash))
    pred=np.zeros(n);baseline=np.zeros(n);ids=np.zeros(n,dtype=int)
    for f,(tr,va) in enumerate(splits):
        ids[va]=f;pred[va]=np.clip(clone(base).fit(sets[feature][tr],y[tr]).predict(sets[feature][va]),0,5)
        values=[]
        for est in [old,hist]:
            model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),clone(est)).fit(sets['all'][tr],y[tr]);values.append(model.predict(sets['all'][va]))
        baseline[va]=np.clip(.75*values[0]+.25*values[1],0,5)
        print(json.dumps({'fold':f,'base':metrics(y[va],pred[va]),'baseline':metrics(y[va],baseline[va])}),flush=True)
    full=clone(base).fit(sets[feature][:n],y);joblib.dump(full,OUT/'prepared_base.joblib')
    np.savez(OUT/'prepared_secondary.npz',base_model=name,fold_ids=ids,target=y,base_oof=pred,baseline_oof=baseline,base_train=np.clip(full.predict(sets[feature][:n]),0,5),base_test=np.clip(full.predict(sets[feature][n:]),0,5))
    print('Prepared',name,metrics(y,pred),metrics(y,baseline),flush=True)
if __name__=='__main__':main()
