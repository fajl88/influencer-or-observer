# 🤖 Copilot Instructions - Influencer or Observer Kaggle Competition

## 📋 Project Overview

Binary classification of French tweets to predict user social roles: **Influencer** (1) vs **Observer** (0).
- **Metric**: Accuracy
- **Train**: 154,914 tweets / 38,560 users
- **Test**: 103,380 tweets / 25,890 users
- **Target**: 0.89+ accuracy on Kaggle leaderboard

## 🏗️ Architecture Overview

```
├── notebooks/
│   └── transformer_ultimate.ipynb    # 🎯 MAIN: Multi-seed Transformer + XGB ensemble
├── preprocessing_improved.ipynb       # CamemBERT embeddings extraction
├── feature_engineering.ipynb          # Structured features (45+ features)
├── ensemble_stacking.ipynb            # LightGBM/XGBoost/NN stacking
├── data/
│   ├── train.jsonl, kaggle_test.jsonl # Raw JSONL data
│   ├── features/X_train_features.npy  # Pre-computed engineered features
│   └── embeddings/                    # CamemBERT embeddings cache
├── submission/                        # All CSV submissions go here
└── models/                            # Saved model artifacts
```

## ⚠️ Critical Data Constraints

**MISSING COLUMNS** (do NOT reference):
- `user.followers_count` - DOES NOT EXIST
- `user.friends_count` - DOES NOT EXIST  
- `user.verified` - DOES NOT EXIST

**AVAILABLE user columns** (use these instead):
- `user.listed_count` ⭐ (best proxy for influence)
- `user.statuses_count`, `user.favourites_count`
- `user.url`, `user.profile_banner_url`, `user.location`, `user.description`

## 🔥 Best Practices for This Project

### Transformer Fine-tuning
```python
# Use cardiffnlp/twitter-xlm-roberta-base for French tweets
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base"
MAX_LEN = 192  # Increased from 128
EPOCHS = 4
LR = 2e-5
BATCH_SIZE = 16
GRAD_ACC = 2  # Effective batch = 32

# CRITICAL: Use StratifiedGroupKFold to prevent user leakage
from sklearn.model_selection import StratifiedGroupKFold
skf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
# Split by pseudo_user_id (hash of user description + profile_image + statuses)
```

### Multi-Seed Ensemble
```python
SEEDS = [42, 1234]  # Train with multiple seeds
# Average probabilities BEFORE threshold optimization
transformer_oof = np.mean([oof_seed1, oof_seed2], axis=0)
```

### XGBoost GPU Settings
```python
XGBClassifier(
    max_depth=10,
    n_estimators=1500,
    learning_rate=0.025,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    device="cuda",
    early_stopping_rounds=100,
)
```

### Meta-Ensemble (Best Results)
```python
# Stack Transformer + XGBoost OOF probabilities
stack_X = np.vstack([transformer_oof, xgb_oof]).T
meta = LogisticRegression(C=1.0)
meta.fit(stack_X, labels)
# Optimize threshold on meta_oof, NOT on individual models
```

### Threshold Calibration
```python
# Always search for optimal threshold on OOF predictions
thresholds = np.linspace(0.35, 0.65, 31)
for thr in thresholds:
    acc = accuracy_score(labels, (oof_probs >= thr).astype(int))
# Typical best threshold: 0.50-0.58
```

## 🎯 Top Discriminative Features

| Feature | Importance | Notes |
|---------|-----------|-------|
| `user_description_length` | ⭐⭐⭐ | Influencers have longer bios |
| `log_user_listed` | ⭐⭐⭐ | Best proxy for follower count |
| `tweets_per_favourites` | ⭐⭐ | Activity ratio |
| `source_device` (bot detection) | ⭐⭐ | Bots often = Influencers |
| `is_reply` | ⭐ | Observers reply more |

## 🛠️ Development Workflow

```bash
# 1. Activate environment
source kaggle-env/bin/activate

# 2. Run notebooks in order:
#    preprocessing_improved.ipynb → embeddings (skip if cached)
#    feature_engineering.ipynb → features/*.npy
#    notebooks/transformer_ultimate.ipynb → submissions

# 3. Submit to Kaggle
kaggle competitions submit -c influencer-or-observer \
    -f submission/submission_meta_ultimate.csv \
    -m "Multi-seed TF + XGB meta-LR"
```

## 🚨 Common Pitfalls

1. **CV/LB Gap**: If OOF accuracy >> Kaggle score, you have user leakage. Use `StratifiedGroupKFold` with pseudo user IDs.

2. **Memory Issues**: RTX 4000 has 20GB. Use:
   - `fp16=True` for training
   - `BATCH_SIZE=16` with `GRAD_ACC=2`
   - Clear cache: `torch.cuda.empty_cache(); gc.collect()`

3. **Wrong Threshold**: Default 0.5 may not be optimal. Always calibrate on OOF.

4. **Embedding Regeneration**: Takes 2-3 hours on GPU. Use cached `data/embeddings/` files.

## 📊 Expected Performance

| Method | CV Accuracy | Kaggle LB |
|--------|------------|-----------|
| LightGBM (features only) | ~84% | ~83% |
| Transformer (single seed) | ~85% | ~84-85% |
| Multi-seed TF + XGB Meta | ~86-87% | ~85-87% |
| + Pseudo-labeling | ~87-88% | ~86-89% |

## 📁 Key File Locations

- **Features**: `data/features/X_train_features.npy`, `X_kaggle_features.npy`
- **Labels**: `data/y_train.npy`
- **Submissions**: `submission/*.csv` (always save here)
- **Models**: `models/` for checkpoints
