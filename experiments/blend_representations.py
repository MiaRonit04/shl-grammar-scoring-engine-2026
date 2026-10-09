"""Select a small development blend grid, then check a second split before export."""
import argparse
import json
from itertools import combinations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from evaluate_representation import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-predictions',type=Path)
    parser.add_argument('--reference-name',default='reference_v2')
    parser.add_argument('--filename',default='Ronit_Mia_v3.csv')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    folds = pd.read_csv(args.metadata/'folds.csv')
    y = folds.target.to_numpy()
    old_oof = pd.read_csv(args.metadata/'candidate_oof.csv')
    old_secondary = pd.read_csv(args.metadata/'secondary_oof.csv')
    old_test = pd.read_csv(args.metadata/'Ronit_Mia_v2.csv')
    test = pd.read_csv(args.metadata/'test_transcripts.csv')
    assert old_oof.filename.tolist() == old_secondary.filename.tolist() == folds.filename.tolist()
    assert old_test.filename.tolist() == test.filename.tolist()
    prepared = np.load(args.metadata/'prepared_secondary.npz')
    assert np.array_equal(prepared['target'], y)
    speech = np.load(args.metadata/'speech_embeddings.npy')
    sx = np.c_[speech[:, :768], speech[:, 2304:3072]]
    speech_model = joblib.load(args.metadata/'final_speech.joblib')
    reference = {'oof': old_oof.prediction.to_numpy(), 'secondary': old_secondary.candidate.to_numpy(),
                 'test': old_test.label.to_numpy(),
                 'train': .25*prepared['base_train']+.75*np.clip(speech_model.predict(sx[:len(y)]),0,5)}
    if args.reference_predictions:
        loaded=np.load(args.reference_predictions)
        reference={key:loaded[key] for key in reference}
        assert all(reference[key].shape==(len(test) if key=='test' else len(y),) for key in reference)
        assert all(np.isfinite(value).all() for value in reference.values())
    candidates = {args.reference_name: reference}
    provenance = {}
    components = {}
    for path in args.candidate:
        summary = json.loads((path/'summary.json').read_text())
        z = np.load(path/'selected.npz')
        label = path.name
        provenance[label] = summary
        candidate = {'oof': z['oof'], 'secondary': z['secondary_oof'], 'train': z['train'],
                     'test': .5*z['full_test']+.5*z['bagged_test']}
        components[label] = candidate
        for weight in [.25,.5,.75,1.0]:
            name = f'{label}_weight{weight}'
            candidates[name] = {key:(1-weight)*reference[key]+weight*candidate[key] for key in reference}
    # Four fixed, coarse blends per pair; no continuous weight optimizer.
    for (label_a,a),(label_b,b) in combinations(components.items(),2):
        for wa,wb in [(.5,.25),(.25,.5),(.5,.5),(.25,.25)]:
            name=f'{label_a}_{wa}+{label_b}_{wb}+reference_{1-wa-wb}'
            candidates[name]={key:wa*a[key]+wb*b[key]+(1-wa-wb)*reference[key] for key in reference}
    results = [{'name': name, 'development': metrics(y, values['oof'])} for name, values in candidates.items()]
    results.sort(key=lambda r:r['development']['rmse'])
    selected = results[0]
    pred = candidates[selected['name']]
    selected['secondary'] = metrics(y, pred['secondary'])
    selected['training'] = metrics(y, pred['train'])
    baseline = {'development': metrics(y, reference['oof']), 'secondary': metrics(y, reference['secondary'])}
    passed = (selected['development']['rmse'] < baseline['development']['rmse']-.002
              and selected['secondary']['rmse'] < baseline['secondary']['rmse']-.002)
    summary = {'selected': selected, 'baseline': baseline, 'reference_name':args.reference_name,'passed_export_gate': passed,
               'development_comparison': results, 'components': provenance,
               'limitation': 'Development folds are reused for selection; second split is a stability check, not an untouched holdout.'}
    (args.output/'blend_summary.json').write_text(json.dumps(summary, indent=2))
    if passed:
        submission = pd.DataFrame({'filename': test.filename, 'label': pred['test']})
        assert len(submission)==216 and submission.filename.is_unique
        assert np.isfinite(submission.label).all() and submission.label.between(0,5).all()
        submission.to_csv(args.output/args.filename, index=False)
        np.savez(args.output/'selected_predictions.npz', **pred)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
