# Influencer or Observer Classification

A machine learning project for classifying Twitter users as **Influencers** or **Observers** using an ensemble of neural networks and gradient boosting models.

## Project Structure

```
influencer-or-observer/
├── code/
│   ├── preprocessing_improved.ipynb    # Data preprocessing & embedding generation
│   └── ensemble_model.ipynb            # Model training & submission generation
├── data/
│   ├── train.jsonl                     # Original training data (required)
│   ├── kaggle_test.jsonl               # Original test data (required)
│   └── embeddings/                     # Generated embeddings (created by preprocessing)
├── models/                             # Saved model checkpoints
├── submission/                         # Generated submission files
└── requirements.txt                    # Python dependencies
```

## Expected Data Files

Place the original competition data files in the `data/` directory:

| File | Description |
|------|-------------|
| `data/train.jsonl` | Training data with labeled tweets (Influencer/Observer) |
| `data/kaggle_test.jsonl` | Test data for Kaggle submission |

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: PyTorch, Transformers (CamemBERT), XGBoost, LightGBM, scikit-learn, Optuna.

## How to Run

### Step 1: Preprocessing (Generate Embeddings)

Run `code/preprocessing_improved.ipynb` to:
- Load raw JSONL data
- Extract structured features (engagement metrics, user profile info, etc.)
- Generate CamemBERT embeddings for tweet text and user descriptions

This creates the following files in `data/`:
- `train_features.csv` and `test_features.csv` (structured features)
- `y_train.npy` (training labels)
- `embeddings/X_train_text_multilayer.npy` (tweet embeddings)
- `embeddings/X_train_desc_multilayer.npy` (user description embeddings)
- `embeddings/X_kaggle_text_multilayer.npy` and `X_kaggle_desc_multilayer.npy` (test embeddings)

### Step 2: Model Training & Submission

Run `code/ensemble_model.ipynb` to:
1. Load preprocessed data and embeddings
2. Train a neural network classifier
3. Train XGBoost and LightGBM models
4. Create a weighted ensemble
5. Generate Kaggle submission files

Submissions are saved to `submission/`:
- `submission_ensemble_model.csv` (ensemble predictions)
- `submission_neural_network_solo.csv` (neural network only)

## Configuration Options

In `ensemble_model.ipynb`, you can configure:

```python
# Model architecture: "simple" (FC) or "multibranch" (separate branches)
MODEL_ARCHITECTURE = "simple"

# Training flags (set to False to load saved models)
TRAIN_NEURAL_NETWORK = True
TRAIN_XGBOOST = False
TRAIN_LIGHTGBM = False
```
