"""Secondary split stability check and full-data refit.
This is a robustness check, not an untouched holdout: labels informed development.
"""
from improve_models import *
from evaluate_speech import speech_specs
from sklearn.model_selection import StratifiedGroupKFold
import joblib, shutil

def base_estimator(name):
    if '_cat_d' in name:
        feature,d=name.split('_cat_d');depth=int(d)
        model=CatBoostRegressor(iterations=1600 if depth>=5 else 1200,depth=depth,learning_rate=.035,l2_leaf_reg=15 if depth==6 else 10,loss_function='RMSE',random_seed=42,verbose=False,thread_count=4,allow_writing_files=False)
    elif '_svr_C' in name:
        feature,params=name.split('_svr_C');c,g=params.split('_g')
        model=SVR(C=float(c),gamma=g if g=='scale' else float(g),epsilon=.1)
    elif name.endswith('_extra'):
        feature=name[:-6];model=ExtraTreesRegressor(n_estimators=600,min_samples_leaf=2,max_features=.8,n_jobs=4,random_state=42)
    else:
        raise ValueError(name)
    return feature,make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),model)

def main():
    folds,y,sets,te=load();n=len(y)
    cfg=json.loads((OUT/'candidate_summary.json').read_text());w=cfg['selected']['speech_weight']
    feature,base=base_estimator(cfg['base_model'])
    speech_data=np.load(ROOT/'experiments/speech_embeddings.npy')
    speech_x,speech=speech_specs(speech_data,sets['basic'])[cfg['speech_model']]
    def pipeline(est):
        return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),clone(est))
    old=CatBoostRegressor(iterations=400,depth=4,learning_rate=.035,l2_leaf_reg=10,loss_function='RMSE',random_seed=42,verbose=False,thread_count=4,allow_writing_files=False)
    hist=HistGradientBoostingRegressor(max_iter=150,learning_rate=.04,max_leaf_nodes=10,l2_regularization=8,early_stopping=False,random_state=42)
    bins=pd.qcut(y,5,labels=False,duplicates='drop')
    splits=list(StratifiedGroupKFold(5,shuffle=True,random_state=2026).split(np.zeros(n),bins,folds.audio_hash))
    prepared=None
    if os.environ.get('SHL_USE_PREPARED_BASE')=='1':
        prepared=np.load(OUT/'prepared_secondary.npz')
        assert str(prepared['base_model'])==cfg['base_model']
        assert np.array_equal(prepared['target'],y)
    pred=np.zeros(n);baseline=np.zeros(n);perfold=[]
    for f,(tr,va) in enumerate(splits):
        if prepared is not None:
            assert np.all(prepared['fold_ids'][va]==f)
            bp=prepared['base_oof'][va]
        else:
            bp=clone(base).fit(sets[feature][tr],y[tr]).predict(sets[feature][va])
        sp=clone(speech).fit(speech_x[tr],y[tr]).predict(speech_x[va])
        pred[va]=(1-w)*np.clip(bp,0,5)+w*np.clip(sp,0,5)
        if prepared is not None:
            baseline[va]=prepared['baseline_oof'][va]
        else:
            bp=pipeline(old).fit(sets['all'][tr],y[tr]).predict(sets['all'][va])
            hp=pipeline(hist).fit(sets['all'][tr],y[tr]).predict(sets['all'][va])
            baseline[va]=np.clip(.75*bp+.25*hp,0,5)
        row={'fold':f,'candidate':metrics(y[va],pred[va]),'baseline_architecture_uncalibrated':metrics(y[va],baseline[va])}
        perfold.append(row);print(json.dumps(row),flush=True)
    train=[];test=[]
    for tag,pipe,x in [('base',base,sets[feature]),('speech',speech,speech_x)]:
        if tag=='base' and prepared is not None:
            train.append(prepared['base_train']);test.append(prepared['base_test'])
            shutil.copyfile(OUT/'prepared_base.joblib',OUT/'final_base.joblib')
            continue
        model=clone(pipe).fit(x[:n],y)
        train.append(np.clip(model.predict(x[:n]),0,5));test.append(np.clip(model.predict(x[n:]),0,5))
        joblib.dump(model,OUT/f'final_{tag}.joblib')
    full_train=(1-w)*train[0]+w*train[1];full_test=(1-w)*test[0]+w*test[1]
    # Full-data fit and CV bagging combine data coverage with variance reduction.
    b=np.load(OUT/f"{cfg['base_model']}.npz")['test'];s=np.load(OUT/f"{cfg['speech_model']}.npz")['test']
    bag_test=(1-w)*b+w*s
    submission=pd.DataFrame({'filename':te.filename,'label':.5*bag_test+.5*full_test})
    assert len(submission)==216 and submission.filename.is_unique
    assert np.isfinite(submission.label).all() and submission.label.between(0,5).all()
    submission.to_csv(OUT/'Ronit_Mia_v2.csv',index=False)
    pd.DataFrame({'filename':folds.filename,'target':y,'candidate':pred,'baseline':baseline}).to_csv(OUT/'secondary_oof.csv',index=False)
    result={'secondary_seed':2026,'candidate':metrics(y,pred),'baseline_architecture_uncalibrated':metrics(y,baseline),'folds':perfold,'full_fit_training':metrics(y,full_train),'test_prediction_strategy':'50% full-data model + 50% original five-fold ensemble','limitation':'Secondary split is a robustness check, not untouched external validation.'}
    (OUT/'secondary_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
