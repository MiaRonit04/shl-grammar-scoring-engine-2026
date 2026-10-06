"""Export saved predictions after resolving the supplied template's ID mismatch."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--template',type=Path,help='Correct competition sample_submission.csv')
parser.add_argument('--use-test-order',action='store_true',help='Explicitly authorize all test IDs in test.csv order instead of incompatible template IDs')
parser.add_argument('--output',type=Path,default=Path('artifacts/submission.csv'))
args=parser.parse_args()
config=json.loads(Path('artifacts/run_config.json').read_text())
pred=pd.read_csv('artifacts/test_predictions.csv')
template_path=args.template or Path(config['dataset'])/'sample_submission.csv'
template=pd.read_csv(template_path)
compatible=len(template)==len(pred) and set(template.iloc[:,0])==set(pred.iloc[:,0])
id_col,score_col=pred.columns
assert template.columns.tolist()==pred.columns.tolist(),'Template schema differs from saved predictions'
assert not pred[id_col].duplicated().any() and pred[id_col].notna().all()
assert not template[id_col].duplicated().any() and template[id_col].notna().all()
if len(template)==len(pred) and set(template[id_col])==set(pred[id_col]):
    result=template.copy()
    result[score_col]=result[id_col].map(pred.set_index(id_col)[score_col])
elif args.use_test_order:
    result=pred.copy()
else:
    parser.error('Template IDs/row count do not match test predictions. Provide a corrected --template, or explicitly authorize --use-test-order.')
assert len(result)==len(pred)
assert np.isfinite(result[score_col]).all() and result[score_col].between(0,5).all()
args.output.parent.mkdir(parents=True,exist_ok=True)
result.to_csv(args.output,index=False)
Path('artifacts/submission_export.json').write_text(json.dumps({'path':str(args.output.resolve()),'template':str(template_path.resolve()),'template_compatible':compatible,'explicit_test_order':args.use_test_order},indent=2))
print(f'Wrote {len(result)} predictions to {args.output.resolve()}')
