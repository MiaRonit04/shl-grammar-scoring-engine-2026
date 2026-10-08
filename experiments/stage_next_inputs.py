"""Stage private baseline artifacts for subsequent reproducible experiments."""
import argparse
import shutil
from pathlib import Path
import numpy as np
from improve_models import load


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    files={
        'folds.csv':'artifacts/folds.csv',
        'audio_audit.csv':'artifacts/audio_audit.csv',
        'train_transcripts.csv':'artifacts/train_transcripts.csv',
        'test_transcripts.csv':'artifacts/test_transcripts.csv',
        'candidate_oof.csv':'experiments/results/candidate_oof.csv',
        'secondary_oof.csv':'experiments/results/secondary_oof.csv',
        'prepared_secondary.npz':'experiments/results/prepared_secondary.npz',
        'final_speech.joblib':'experiments/results/final_speech.joblib',
        'speech_embeddings.npy':'experiments/speech_embeddings.npy',
        'Ronit_Mia_v2.csv':'experiments/results/Ronit_Mia_v2.csv',
    }
    for name,relative in files.items():
        source=root/relative
        if not source.exists():
            raise FileNotFoundError(f'{relative}: first complete the baseline improvement pipeline, including prepare_secondary_base.py')
        shutil.copyfile(source,args.output/name)
    _,_,sets,_=load()
    np.save(args.output/'extended.npy',sets['extended'])
    print('Private inputs staged; do not publish this directory.',flush=True)


if __name__=='__main__':main()
