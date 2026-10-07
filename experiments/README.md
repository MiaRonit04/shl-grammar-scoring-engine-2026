# Grammar-scoring improvement experiments

The original submitted predictions remain in `artifacts/Ronit_Mia.csv`. Their public Kaggle score was 0.4912 (observed on 7 October 2026). The overview describes a custom metric involving RMSE and Pearson correlation without specifying its exact formula there. Local metrics are reported separately.

These experiments use only the provided training labels and competition recordings, plus a pretrained speech model. Test labels, manual test annotations and leaderboard feedback are not used to fit or select models. Recordings are processed locally.

## Reproduce

Run the original `grammar_scoring_engine.ipynb` first to generate transcripts, linguistic/acoustic features, MiniLM embeddings, the audio audit and the original fold assignments. From the repository root, run these scripts in order:

1. `python experiments/extract_acoustics.py`
2. `python experiments/improve_models.py`
3. `python experiments/extract_speech.py`
4. `python experiments/evaluate_speech.py`
5. `python experiments/select_candidate.py`
6. `python experiments/validate_candidate.py`
7. `python experiments/build_improvement_report.py`

Use a fresh environment with the project's requirements. `results/runtime.json` records the actual numerical-library versions. The original local environment had an inconsistent PyTorch installation, so the completed improvement run uses an isolated installation of the same Torch and Transformers versions.

## Models and validation

The acoustic extension adds 522 fixed spectral and temporal statistics: log-mel and MFCC distributions, MFCC differences, energy, spectral entropy/flatness, activity and duration. It fits no target-dependent parameters.

The frozen Apache-2.0 model `facebook/wav2vec2-base-960h` is pinned to revision `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`. It processes all audio in ten-second chunks. Layers 6, 9 and 12 are pooled with frame-weighted means and standard deviations, producing 4,608 features per recording. The CTC output head is not used; the training-time mask embedding is also unused in evaluation mode.

The search evaluates feature ablations, support-vector regression, Ridge, CatBoost and Extra Trees. Joint speech/handcrafted SVR candidates balance feature blocks after fold-fitted scaling. Joint tree candidates use 32 speech principal components, fitted inside each fold. All imputation and learned preprocessing remain inside the training folds. Identical audio hashes stay in the same fold.

The selection stage compares a small convex blend grid using development OOF predictions. A second split (seed 2026) checks stability against a baseline architecture. It is not an untouched holdout: selecting models on reused development labels can make estimates optimistic. Unknown speaker overlap is another limitation. No validation result guarantees a leaderboard position.

Final test predictions average full-data fitting and five-fold bagging. The exporter verifies all 216 official test identifiers, their order, and finite predictions in [0, 5]. The submission file is `results/Ronit_Mia_v2.csv`.

## Local acceleration and privacy

Optional `GRAMMAR_AUDIO_ROOT` points to a staged directory containing `train/` and `test/`. Each staged recording is checked against the audio audit's SHA-256 before use. `GRAMMAR_SPEECH_CACHE` optionally relocates the per-record embedding cache. These settings change storage locations, not model inputs.

`prepare_secondary_base.py` is an optional acceleration for the current run. If it was generated from the same features and code, `SHL_USE_PREPARED_BASE=1` lets the validation script reuse it after checking the model name, targets and fold IDs. The default validation recomputes the comparison.

Feature matrices, transcripts, recordings, per-record training predictions, model weights and caches must stay local. Publish only source, aggregate results, the report/notebook and the selected test prediction CSV.
