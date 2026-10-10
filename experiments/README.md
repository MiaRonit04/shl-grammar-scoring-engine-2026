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

## Experiments following the v3 submission

The verified submitted model remains v3. Additional W2v-BERT, kernel-ridge and PCA experiments did not pass the preset export gate against v3; their aggregate results are in `results/round4/`. A more complex representation is retained only if it improves measured validation.

`extract_wavlm_frames.py` caches frozen layer-12 frame features. It verifies that the truncated encoder produces the same layer output as the full model. `evaluate_attentive_pooling.py` trains a small regression network with attentive mean/std pooling, inspired by [Okabe et al.](https://arxiv.org/abs/1803.10963). Normalization is fitted on each training fold. Two predetermined training durations and two random seeds are compared, followed by secondary validation. These experiments completed without passing the export threshold against v3. Their aggregate results are recorded in `results/round4/`; no new leaderboard score is claimed.

After reproducing v3 and staging its private metadata, run:

```bash
python experiments/extract_wavlm_frames.py --audio-root "$GRAMMAR_AUDIO_ROOT" --output experiments/private_round4/frames
python experiments/evaluate_attentive_pooling.py --features experiments/private_round4/frames --metadata experiments/private_round3/metadata --output experiments/private_round4/attentive
python experiments/blend_representations.py --metadata experiments/private_round3/metadata --candidate experiments/private_round4/attentive --reference-predictions experiments/private_round3/blend/selected_predictions.npz --reference-name reference_v3 --filename Ronit_Mia_v4.csv --output experiments/private_round4/attentive_blend
python -m unittest discover -s experiments -p test_attentive_pooling.py -v
```

CUDA, Apple MPS and CPU are detected automatically. Fixed speech features can be reused; learned normalization and the scoring network are refitted inside every fold. Use a fresh results directory after changing source or inputs because cached predictions are fingerprinted. The tests check padding invariance and isolation from held-out labels and normalization statistics.

`extract_acceptability.py` separately computes frozen [RoBERTa CoLA](https://huggingface.co/textattack/roberta-base-CoLA) sentence-acceptability features from existing local transcripts. It downloads a pinned pretrained model, not the CoLA dataset. Its class probabilities are features, not calibrated grammar scores. Evaluate them with `evaluate_language_likelihood.py --feature-file acceptability.npy --feature-name acceptability` alongside the required metadata/language/output directory arguments. `evaluate_representation.py` can evaluate the pooled embeddings from the same output directory. ASR correction of errors and the difference between written and spoken language remain limitations.

For all new candidates, the exporter compares against the exact v3 reference and requires an RMSE improvement greater than 0.002 on both development and secondary splits. Repeated selection still introduces optimism; secondary validation does not replace an untouched holdout or an actual scored submission.

The follow-up `extract_wavlm_blocks.py` / `finetune_wavlm_blocks.py` experiment trained encoder layers 11–12, with layers 1–10 frozen and cached separately. Each scoring head was initialized using only its own fold’s training rows. The fixed two-/four-epoch comparison did not improve secondary validation. Its reconstruction and finite-gradient pilot passed, but the final candidate was rejected.
