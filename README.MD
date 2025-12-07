# Influencer or Observer

Short project notebook pipeline for classifying Twitter posts as "influencer" vs "observer" (binary classification). The repository contains preprocessing and modeling notebooks that build features from raw JSONL tweets, compute CamemBERT text embeddings, combine those with structured features, train models and generate a Kaggle-style submission CSV.

**Repository layout**
- `preprocessing.ipynb`: data loading, cleaning, structured feature engineering, and extraction of CamemBERT text embeddings (single-layer and multi-layer options). Produces processed numpy arrays.
- `model.ipynb`: builds and compares model pipelines (XGBoost primary), performs GridSearchCV, evaluates on a validation split and generates `submission/submission.csv`.
- `data/`: processed data, embeddings and raw input files.
  - `train.jsonl` — raw labeled training examples (JSONL)
  - `kaggle_test.jsonl` — test set for inference
  - `y_train.npy` — labels saved during preprocessing
  - `X_train_processed.npy`, `X_kaggle_processed.npy` — final single-layer feature arrays (structured + 768-dim embeddings)
  - `X_train_processed_multilayer.npy`, `X_kaggle_processed_multilayer.npy` — final multi-layer feature arrays (structured + 2304-dim embeddings)
  - `embeddings/` — saved text embedding arrays
- `submission/submission.csv`: example submission file produced by `model.ipynb`.
- `requirements.txt`: Python dependencies used by notebooks.

**High-level pipeline**
1. Preprocess raw JSONL data and extract features using `preprocessing.ipynb`.
   - Drops list/dict columns, synchronizes train / test columns, fills numeric NaNs and one-hot encodes selected categorical features.
   - Extracts text from `text` or `extended_tweet.full_text`.
   - Optionally computes CamemBERT embeddings (single-layer or multi-layer). Embeddings are computationally intensive and recommended on GPU.
   - Concatenates structured features and text embeddings and saves final arrays in `data/`.
2. Train and evaluate models with `model.ipynb`.
   - Loads the processed arrays, splits training data for a validation check, and runs GridSearchCV for model selection.
   - Implementations: an XGBoost pipeline (primary), optional Logistic Regression and a PyTorch MLP wrapper are present in the notebook (XGBoost is active by default in the notebook).
   - Produces a `submission/submission.csv` containing `ID` and `Prediction` for the test set.


**Important flags & notes**
- Embedding extraction: `preprocessing.ipynb` includes code to compute CamemBERT embeddings. The notebook has a clear warning comment: do not execute the embedding cells unless you need to (they are slow and require significant memory/time). Use a GPU when possible.
- Single-layer vs Multi-layer embeddings: `model.ipynb` contains a `USE_MULTILAYER` flag. When `False` it uses single-layer (layer 12) 768-dim embeddings (total features ~ 776). When `True` it uses the concatenation of layers 6, 9 and 12 (3×768 = 2304 dims) giving ~2312 total features.
- If using the notebooks interactively, inspect and set `DATA_DIR` and file paths at the top of the notebooks if your environment differs.