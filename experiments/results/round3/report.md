# Third-round speech scoring experiments

The selected candidate is `shl-wavlm-results_0.5+shl-whisper-results_0.25+reference_0.25`.

- Development OOF RMSE: **0.505064**; Pearson: **0.916977**.
- Secondary split RMSE: **0.510248**; Pearson: **0.914831**.
- Full-fit training RMSE: **0.078528**; Pearson: **0.998628** (in-sample).
- Previous ensemble development/secondary RMSE: **0.555232 / 0.554599**.

## Method and selection

We extracted frozen [WavLM large](https://huggingface.co/microsoft/wavlm-large) hidden layers 6, 12, 18 and 24 from all provided recordings. Ten-second chunks cover the full recording; frame-weighted means and standard deviations yield fixed features. Standardization and SVR/Ridge regressors are fitted separately inside each training fold. Audio-hash groups and the previous development folds are retained.

We also extracted frozen [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) encoder layers 8, 16, 24 and 32. All audio is processed in 30-second chunks; pooling excludes padded frames. The encoder ran in float16 on the local GPU.

We also tested frozen [GPT-2](https://huggingface.co/openai-community/gpt2) transcript embeddings and token-likelihood statistics. GPT-2 embeddings and mean-only Wav2Vec2 pooling did not beat the previous ensemble in the tested blend grid. Likelihood statistics improved the handcrafted CatBoost component. The final comparison uses a small fixed convex-blend grid, then checks the selected blend on the secondary split. Test predictions average full-data fitting and fold bagging for each new component.

Only competition recordings/transcripts and provided training labels are used. Pretrained models are pinned by revision. Audio and text inference run locally. No manual test labels or leaderboard-derived labels are used. The CSV follows all 216 official test IDs and contains finite predictions between 0 and 5.

## Limits

These are local development results, not Kaggle scores. The models and blends were selected using reused development folds, so selection optimism is possible. The second split is a stability check, not an untouched holdout. Speaker overlap may remain because speaker IDs are unavailable. No local score establishes a leaderboard rank. The last verified public score before this round was 0.4337.

## Reproduction

Complete the original notebook and improvement pipeline first, including `prepare_secondary_base.py` and `validate_candidate.py`. Run `stage_next_inputs.py` to assemble the private intermediate inputs. Extract WavLM and GPT-2 features, evaluate them, then run `blend_representations.py` and this report generator. The companion notebook contains the command sequence. Intermediate metadata, transcripts, embeddings and per-record training predictions must remain private.

## Verified Kaggle result

Public score: **0.4017**. Observed rank: **113**, on 2026-10-09. Leading score at that observation: **0.3064**. Rank #1 has not been established unless explicitly verified.
