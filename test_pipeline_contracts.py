"""Focused tests of leakage boundaries, text handling and submission alignment."""
import ast,json,tempfile,warnings
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold,StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

nb=json.loads(Path('grammar_scoring_engine.ipynb').read_text())
ns=dict(globals())
for cell in nb['cells']:
    if cell['cell_type']!='code': continue
    tree=ast.parse(''.join(cell['source']))
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in {'make_folds','numeric_pipeline','normalize_text'}:
            exec(compile(ast.Module(body=[node],type_ignores=[]),'<notebook>','exec'),ns)
import re
ns['re']=re

class Contracts(unittest.TestCase):
    def test_group_folds_cover_once_without_duplicate_leakage(self):
        groups=np.repeat(np.arange(30),2); y=np.repeat(np.tile([1,2,3,4,5],6),2)
        folds=ns['make_folds'](y,groups,5,42)
        indices=[]
        for tr,va in folds:
            self.assertFalse(set(groups[tr])&set(groups[va])); indices.extend(va)
        self.assertEqual(sorted(indices),list(range(len(y))))
    def test_preprocessing_is_fit_on_training_only(self):
        X=np.array([[0.,np.nan],[1.,1.],[2.,2.],[3.,3.]])
        p=ns['numeric_pipeline'](Ridge()).fit(X,np.arange(4.))
        before=p.named_steps['impute'].statistics_.copy()
        p.predict(np.array([[1e9,np.nan]]))
        np.testing.assert_array_equal(before,p.named_steps['impute'].statistics_)
        self.assertEqual(before[0],1.5)
    def test_feature_assembly_supports_readonly_pandas_arrays(self):
        context={'pd':pd,'np':np,'all_text':pd.DataFrame({c:[0.,np.nan,1.,2.,3.,4.,5.,6.] for c in ['n_segments','avg_logprob','no_speech_prob','segment_gap_mean','asr_failed','empty_transcript']}),
                 'ling':pd.DataFrame({'words':np.arange(8.)}),'emb':np.ones((8,4)),
                 'acoust':pd.DataFrame({'duration':np.arange(8.)}),
                 'USE_LINGUISTIC_FEATURES':True,'USE_AUDIO_FEATURES':True,
                 'train_df':pd.DataFrame({'id':range(5)}),'test_df':pd.DataFrame({'id':range(3)}),'y':np.ones(5)}
        source=next(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code' and ''.join(c['source']).startswith('metadata=all_text'))
        exec(source,context)
        for x in context['Xsets'].values():
            self.assertTrue(x.flags.writeable); self.assertEqual(len(x),5)
    def test_optional_native_library_failure_is_nonfatal(self):
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
        import types
        def unavailable(name): raise RuntimeError('Native runtime missing')
        calls=[]
        context={'FAST_MODE':True,'SEED':42,'Xsets':{'B_embeddings':np.zeros((10,2))},
                 'evaluate':lambda *args,**kwargs:calls.append(args[0]),
                 'ExtraTreesRegressor':ExtraTreesRegressor,'RandomForestRegressor':RandomForestRegressor,
                 'HistGradientBoostingRegressor':HistGradientBoostingRegressor,
                 'importlib':types.SimpleNamespace(import_module=unavailable)}
        source=next(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code' and ''.join(c['source']).startswith('tree_count='))
        exec(source,context)
        self.assertEqual(len(calls),3)
    def test_normalization_preserves_repetitions_and_fillers(self):
        self.assertEqual(ns['normalize_text'](' Um,  I I mean...\n he go. '),'Um, I I mean... he go.')
    def run_export(self,sample,test,policy):
        with tempfile.TemporaryDirectory() as directory:
            context={'OUTPUT_DIR':Path(directory),'template_compatible':len(sample)==len(test) and set(sample.filename)==set(test.filename),
                     'sample_df':sample,'test_df':test,'sample_filename_col':'filename','test_filename_col':'filename','pred_col':'label',
                     'test_predictions':pd.DataFrame({'filename':test.filename,'label':np.arange(len(test))+.5}),
                     'SUBMISSION_POLICY':policy,'warnings':warnings,'np':np,'pd':pd,'TARGET_MIN':0,'TARGET_MAX':5,'display':lambda x:None}
            source=next(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code' and ''.join(c['source']).startswith('submission_path='))
            exec(source,context)
            return context['submission'],context['submission_written']
    def test_submission_reorders_by_id_not_position(self):
        test=pd.DataFrame({'filename':['b.wav','a.wav'],'label':[-1,-1]})
        sample=pd.DataFrame({'filename':['a.wav','b.wav'],'label':[-1,-1]})
        out,written=self.run_export(sample,test,'strict')
        self.assertTrue(written); self.assertEqual(out.label.tolist(),[1.5,.5]); self.assertEqual(out.filename.tolist(),sample.filename.tolist())
    def test_incompatible_template_refused(self):
        test=pd.DataFrame({'filename':['b.wav','a.wav']})
        sample=pd.DataFrame({'filename':['x.wav'],'label':[-1]})
        out,written=self.run_export(sample,test,'strict')
        self.assertFalse(written); self.assertIsNone(out)
    def test_authorized_test_order_export(self):
        test=pd.DataFrame({'filename':['b.wav','a.wav']})
        sample=pd.DataFrame({'filename':['x.wav'],'label':[-1]})
        out,written=self.run_export(sample,test,'test_order')
        self.assertTrue(written); self.assertEqual(out.filename.tolist(),test.filename.tolist())

if __name__=='__main__': unittest.main()
