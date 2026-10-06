# Final Report
## Problem and Dataset
Continuous spoken-English grammar scoring in [0, 5], using 769 labeled recordings and 216 test recordings. Typical intended duration is 45–60 seconds; measured median training duration is 60.07 seconds. Test label-like fields were ignored.
## Methodology and Preprocessing
Mono 16 kHz audio → base.en ASR → lightly normalized transcripts → lexical/syntactic features and frozen sentence-transformers/all-MiniLM-L6-v2 embeddings, optionally supplemented with acoustic features. Pauses/repetitions were retained. Long transcripts were chunked. Feature transformations were fitted inside folds. No external labeled data was added.
## Modeling and Validation
Models include mean/Ridge/ElasticNet, TF-IDF, ExtraTrees, RandomForest, histogram boosting, and optional installed boosting libraries. The experiment table below is the source of truth for what ran. 5 common folds were used with target stratification where feasible and exact-audio grouping. PCA, small blends and nested affine calibration were tested.
| Model | RMSE | Pearson |
| --- | --- | --- |
| Blend 75-25 + nested calibration | 0.66085 | 0.84578 |
| Blend 75-25 | 0.67050 | 0.84405 |
| Blend equal3 | 0.67081 | 0.84468 |
| Multimodal catboost | 0.67229 | 0.84341 |
| Blend equal2 | 0.67256 | 0.84233 |
| Multimodal HistGradientBoosting | 0.68801 | 0.83237 |
| D_multimodal ExtraTrees | 0.69062 | 0.83467 |
| Multimodal RandomForest | 0.70077 | 0.82693 |
| D_multimodal Ridge 100 | 0.72699 | 0.80991 |
| audio_only Ridge 100 | 0.84933 | 0.72778 |
| audio_only Ridge 10 | 0.85135 | 0.72664 |
| D_multimodal Ridge 10 | 0.86340 | 0.74144 |
| C_text_combined ExtraTrees | 0.97987 | 0.62882 |
| C_text_combined Ridge 100 | 0.99071 | 0.61798 |
| B_embeddings Ridge 100 | 1.00619 | 0.59867 |
| Embeddings PCA95 Ridge 100 | 1.01205 | 0.58621 |
| Embeddings PCA95 Ridge 10 | 1.03860 | 0.57313 |
| B_embeddings ExtraTrees | 1.04714 | 0.58431 |
| A_linguistic ExtraTrees | 1.05944 | 0.51791 |
| A_linguistic Ridge 100 | 1.07847 | 0.49555 |
| Linguistic ElasticNet | 1.08505 | 0.49100 |
| A_linguistic Ridge 10 | 1.09359 | 0.48410 |
| B_embeddings Ridge 10 | 1.16173 | 0.51660 |
| C_text_combined Ridge 10 | 1.16394 | 0.52184 |
| Transcript TF-IDF Ridge | 1.16996 | 0.56411 |
| Mean baseline | 1.23847 | -0.02793 |
## Final Metrics
- Training RMSE: 0.366269
- Training Pearson: 0.957219
- OOF RMSE: 0.660852
- OOF Pearson: 0.845779

Training metrics are in-sample. OOF scores are more realistic, but selecting among experiments on these same folds adds optimism.
## Interpretation
Selected system: **Blend 75-25 + nested calibration**. Members: [('Multimodal catboost', 0.75), ('Multimodal HistGradientBoosting', 0.25)]. Affine calibration retained: True. Matched Ridge (alpha=100) OOF RMSE by feature set: A_linguistic: 1.0785; B_embeddings: 1.0062; C_text_combined: 0.9907; D_multimodal: 0.7270; audio_only: 0.8493. These comparisons hold the regressor and folds fixed; lower RMSE is better. Final-tree diagnostics: Multimodal catboost: top features avg_logprob, root_ttr, no_speech_prob, mfcc_1_std, unique_words. Native tree importance is descriptive and can favor high-dimensional/correlated groups; it is not a causal measure. Linguistic importance/coefficient plots provide complementary interpretation. Neither acoustic quality nor vocabulary is identical to grammar, and duration/noise shortcuts may fail under dataset shift.
## Limitations
Small sample size; imperfect ASR may erase grammatical mistakes; accent/noise and topic sensitivity; unknown speaker overlap; imperfect segmentation; potential hyperparameter-selection overfitting. Pretrained representations are not grammar-specific. FAST_MODE=True; full-mode quality requires its own validation. A single development run cannot establish leaderboard superiority.
## Submission and Conclusion
Template compatible: False. Submission written: True. Policy: test_order, matching the official 216-row upload requirement. A complete test_predictions.csv is always exported. The selected architecture minimizes development RMSE subject to correlation checks; extra complexity is retained only with measured improvements. The official Kaggle upload dialog requires 216 rows plus a header. A fresh official test.csv was verified byte-for-byte against the local input; prediction IDs and order match. The original 204-row sample template is inconsistent with the current test set. The prediction file has not yet been uploaded or scored on Kaggle.
