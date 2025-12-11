# 📚 Documentation des Notebooks - Influencer or Observer

## 🎯 Vue d'ensemble

Ce projet contient plusieurs notebooks avec différentes approches pour classifier des tweets français en **Influenceur (1)** vs **Observateur (0)**.

---

## 🗺️ CARTE DES PIPELINES

```
                            ┌─────────────────────────────────────┐
                            │     DONNÉES BRUTES                  │
                            │  train.jsonl / kaggle_test.jsonl    │
                            └──────────────┬──────────────────────┘
                                           │
               ┌───────────────────────────┼───────────────────────────┐
               ▼                           ▼                           ▼
┌──────────────────────────┐  ┌──────────────────────────┐  ┌─────────────────────────┐
│ preprocessing_embeddings │  │   preprocessing.ipynb    │  │ feature_engineering     │
│   (EMBEDDINGS ONLY) ⭐   │  │     (version legacy)     │  │  (FEATURES ONLY) ⭐     │
└────────────┬─────────────┘  └──────────────────────────┘  └────────────┬────────────┘
             │                                                           │
             ▼                                                           ▼
┌──────────────────────────┐                              ┌─────────────────────────┐
│ X_train_embeddings.npy   │                              │ X_train_features.npy    │
│ X_kaggle_embeddings.npy  │                              │ X_kaggle_features.npy   │
│ (768d ou 3072d)          │                              │ (45+ features)          │
└────────────┬─────────────┘                              └────────────┬────────────┘
             │                                                         │
             └─────────────────────────┬───────────────────────────────┘
                                       ▼
                        ┌──────────────────────────────────────────┐
                        │        model_with_features.ipynb         │
                        │   (charge les 2, combine, entraîne)      │
                        └──────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              PIPELINE TRANSFORMERS                                       │
│                         (Fine-tuning sur texte brut)                                    │
└─────────────────────────────────────────────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────┐  ┌──────────────────────────┐  ┌─────────────────────────┐
│ transformer_finetune     │  │  transformer_ultimate    │  │ transformer_ultimate_v2 │
│ (version simple CV)      │  │ (multi-seed + XGB meta)  │  │  (version optimisée)    │
└──────────────────────────┘  └──────────────────────────┘  └─────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                          PIPELINE NEURAL NETWORK                                         │
│                   (Réseaux de neurones sur embeddings)                                  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│     train_model.ipynb    │  │  train_model_improved    │
│   (NN basique 3 branches)│  │ (NN + Attention + EMA)   │
└──────────────────────────┘  └──────────────────────────┘
```

---

## 📋 DÉTAIL DE CHAQUE NOTEBOOK

---

### 1️⃣ `preprocessing_embeddings.ipynb` (NOUVEAU - Embeddings uniquement) ⭐

#### 📝 Description
Notebook **simplifié** qui génère **UNIQUEMENT** les embeddings CamemBERT. Pas de feature engineering !

#### 📥 Entrées
- `data/train.jsonl` - Données d'entraînement brutes
- `data/kaggle_test.jsonl` - Données de test brutes

#### 📤 Sorties
- `data/embeddings/X_train_embeddings.npy` - Embeddings (768d ou 3072d multi-layer)
- `data/embeddings/X_kaggle_embeddings.npy`
- `data/y_train.npy` - Labels (0/1)

#### 🔧 Ce qu'il fait
1. Charge les JSONL
2. Extrait le texte complet des tweets
3. Génère des embeddings CamemBERT multi-layer (4 dernières couches concaténées)
4. Sauvegarde les `.npy`

#### ⚙️ Configuration
```python
model_name = "camembert-base"
max_length = 128
use_multi_layer = True  # 4 couches → 3072d
pooling = "mean"        # ou "cls"
```

#### 🔗 Utilisé par
- `model_with_features.ipynb`
- `ensemble_stacking.ipynb`

---

### 2️⃣ `preprocessing.ipynb` (VERSION LEGACY)

#### 📝 Description
Version originale du preprocessing. **Déprécié** - utiliser `preprocessing_embeddings.ipynb` à la place.

#### ⚠️ Problèmes
- Mélange embeddings + features (pas propre)
- Embeddings single-layer uniquement (768d)
- Redondant avec `feature_engineering.ipynb`

---

### 3️⃣ `preprocessing_improved.ipynb` (VERSION LEGACY)

#### 📝 Description
Version améliorée mais **déprécié** - mélange encore embeddings et features.

#### ⚠️ Problèmes
- Contient du feature engineering (redondant avec `feature_engineering.ipynb`)
- Trop complexe

**Remplacé par** : `preprocessing_embeddings.ipynb` (embeddings) + `feature_engineering.ipynb` (features)

---

### 4️⃣ `feature_engineering.ipynb` ⭐⭐⭐

#### 📝 Description
Notebook **standalone** dédié uniquement à l'extraction de features structurées (sans embeddings). Génère ~45+ features discriminantes avec analyse d'importance.

#### 📥 Entrées
- `data/train.jsonl`
- `data/kaggle_test.jsonl`

#### 📤 Sorties
- `data/features/X_train_features.npy` - 45+ features structurées
- `data/features/X_kaggle_features.npy`
- `data/train_features.csv` - Version CSV avec noms de colonnes
- `data/test_features.csv`
- `data/features/preprocessor.joblib` - Pipeline de preprocessing

#### 🔧 Ce qu'il fait
1. **Features textuelles** (sans NLP lourd) :
   - Longueur tweet, word count, uppercase ratio
   - Compteurs hashtags/mentions/URLs/emojis
   - Détection reply, RT/QT, call-to-action
   
2. **Features utilisateur** (⚠️ SANS followers_count/friends_count qui n'existent pas) :
   - `user.listed_count` ⭐ (meilleur proxy d'influence)
   - `user.statuses_count`, `user.favourites_count`
   - `user_description_length`
   - `user_has_url`, `user_has_banner`, `user_has_location`
   
3. **Features structurées tweet** :
   - Source device (iphone, android, web, bot...)
   - Engagement (retweet_count, favorite_count...)
   - Entités (entities.hashtags, entities.urls...)
   
4. **Ratios discriminants** :
   - `listed_per_status` = listed_count / statuses_count
   - `tweets_per_favourites` = statuses / favourites
   
5. **Analyse d'importance** avec LightGBM

#### 📊 Top Features Découvertes
| Feature | Importance LightGBM |
|---------|---------------------|
| `user_description_length` | 736 |
| `tweets_per_favourites` | 699 |
| `user_favourites_count` | 646 |
| `user_statuses_count` | 639 |
| `user_listed_count` | 607 |
| `log_user_listed` | (corr=0.606 avec label) |

#### 🔗 Utilisé par
- `model_with_features.ipynb`
- `ensemble_stacking.ipynb`
- `notebooks/transformer_ultimate.ipynb`
- `notebooks/train_model.ipynb`
- `notebooks/train_model_improved.ipynb`

---

### 4️⃣ `model.ipynb` (BASELINE XGBoost)

#### 📝 Description
Modèle **baseline simple** : XGBoost GPU sur les features combinées (embeddings + quelques features structurées) générées par `preprocessing.ipynb`.

#### 📥 Entrées
- `data/X_train_processed.npy` (ou `X_train_processed_multilayer.npy`)
- `data/X_kaggle_processed.npy`
- `data/y_train.npy`

#### 📤 Sorties
- Prédictions pour submission (dans le notebook)
- Pas de sauvegarde de modèle par défaut

#### 🔧 Ce qu'il fait
1. Charge les features pré-calculées
2. Split train/val
3. Entraîne XGBoost GPU avec GridSearchCV
4. (Optionnel) Compare avec Logistic Regression et MLP PyTorch
5. Génère prédictions

#### 🎯 Performance attendue
- ~83-84% accuracy (limité par features simples)

#### 🔗 Dépend de
- `preprocessing.ipynb` (génère les `.npy`)

---

### 5️⃣ `model_with_features.ipynb` (NN + Boosting + Features) ⭐

#### 📝 Description
Pipeline **complet** qui combine embeddings + features structurées et entraîne 3 modèles.

**IMPORTANT : Ce notebook ne fait PAS de feature engineering !**  
Il charge simplement les données pré-calculées par les notebooks en amont.

#### 📥 Entrées (générées par d'autres notebooks)
| Fichier | Généré par |
|---------|------------|
| `data/embeddings/X_train_multilayer_embeddings.npy` | `preprocessing_improved.ipynb` |
| `data/embeddings/X_kaggle_multilayer_embeddings.npy` | `preprocessing_improved.ipynb` |
| `data/features/X_train_features.npy` | `feature_engineering.ipynb` |
| `data/features/X_kaggle_features.npy` | `feature_engineering.ipynb` |
| `data/y_train.npy` | `preprocessing_improved.ipynb` |

#### 📤 Sorties
- `models/xgb_model.joblib`
- `models/lgbm_model.joblib`
- `models/influencer_model_simple.pt` (NN PyTorch)
- `models/scaler.joblib`
- `submission/*.csv`

#### 🔧 Ce qu'il fait
1. **Chargement données** : 
   - Embeddings depuis `data/embeddings/*.npy`
   - Features depuis `data/features/*.npy`
2. **Combinaison** : `hstack([embeddings, features])` → ~2350 dimensions
3. **Normalisation** : StandardScaler sur les features
4. **3 modèles** :
   - `InfluencerClassifierSimple` (NN PyTorch avec Multi-Sample Dropout)
   - XGBoost GPU
   - LightGBM CPU multi-thread
5. **Configuration flexible** : `TRAIN_NEURAL_NETWORK`, `TRAIN_XGBOOST`, `TRAIN_LIGHTGBM` flags
6. **Ensemble** : moyenne pondérée des 3 modèles

#### ✅ Fonctionnalités
- Chargement/sauvegarde de modèles
- Réentraînement sélectif (ex: seulement le NN)
- Optimisé GPU RTX 4000

#### 🎯 Performance attendue
- ~85-86% accuracy

---

### 6️⃣ `ensemble_stacking.ipynb` (Stacking Avancé) ⭐⭐

#### 📝 Description
Pipeline de **stacking** sophistiqué : LightGBM + XGBoost + NN comme base learners, puis méta-modèle (LogisticRegression) sur les prédictions OOF.

#### 📥 Entrées
- `data/train_features.csv`
- `data/test_features.csv`
- `data/embeddings/X_train_multilayer_embeddings.npy` (optionnel)
- `data/y_train.npy`

#### 📤 Sorties
- `submission/*.csv`
- Prédictions intermédiaires des base learners

#### 🔧 Ce qu'il fait
1. **Charge features** avec normalisation
2. **Définit 4 modèles** :
   - LightGBM (1500 estimators, CPU 20 threads)
   - XGBoost (2000 estimators, GPU)
   - Logistic Regression
   - Neural Network (TweetMLP avec GELU, dropout progressif)
3. **Cross-validation** stratifiée
4. **Stacking** : 
   - Génère OOF predictions de chaque modèle
   - Empile les prédictions
   - Entraîne méta-modèle LogisticRegression sur l'empilement
5. **Optimisation seuil** sur OOF combiné

#### ✅ Avantages
- Combine forces de chaque modèle
- OOF évite le data leakage
- Méta-learner apprend les poids optimaux

#### 🎯 Performance attendue
- ~86-87% accuracy

---

### 7️⃣ `notebooks/transformer_finetune.ipynb` (Transformer Simple)

#### 📝 Description
Fine-tuning **basique** d'un transformer (`cardiffnlp/twitter-xlm-roberta-base`) avec cross-validation stratifiée par pseudo-user.

#### 📥 Entrées
- `data/train.jsonl` (directement, pas de preprocessing)
- `data/kaggle_test.jsonl`

#### 📤 Sorties
- Prédictions dans le notebook
- Pas de sauvegarde par défaut

#### 🔧 Ce qu'il fait
1. **Preprocessing texte** inline :
   - Extrait texte + description + location
   - Construit input enrichi : `{text}[DESC]{description}[META]{device, stats...}`
2. **Pseudo-user grouping** : hash(description + profile_image + statuses) pour éviter leakage
3. **StratifiedGroupKFold** (4 folds)
4. **Fine-tuning** :
   - 3 epochs, batch 16, grad_acc 2
   - fp16, LR 2e-5
5. **Moyenne des folds** pour test
6. **Optimisation threshold** sur OOF

#### 🔧 Différences vs transformer_ultimate
| Aspect | transformer_finetune | transformer_ultimate |
|--------|----------------------|----------------------|
| Seeds | 1 | 4 (multi-seed) |
| Folds | 4 | 5 |
| Epochs | 3 | 5 |
| MAX_LEN | 160 | 224 |
| XGBoost meta | Non | Oui |
| Pseudo-labeling | Non | Oui |

#### 🎯 Performance attendue
- ~84-85% accuracy

---

### 8️⃣ `notebooks/transformer_ultimate.ipynb` (MEILLEUR PIPELINE) ⭐⭐⭐

#### 📝 Description
Pipeline **le plus complet et performant** : multi-seed transformer CV + XGBoost GPU sur features + méta-ensemble avec LogisticRegression + pseudo-labeling.

#### 📥 Entrées
- `data/train.jsonl`
- `data/kaggle_test.jsonl`
- `data/features/X_train_features.npy` (de `feature_engineering.ipynb`)
- `data/features/X_kaggle_features.npy`

#### 📤 Sorties
- `models/transformer_ultimate/transformer_oof.npy`
- `models/transformer_ultimate/transformer_test.npy`
- `submission/submission_meta_ultimate.csv`

#### 🔧 Pipeline complet
1. **Configuration optimisée** :
   - 4 seeds (42, 1234, 2024, 0)
   - 5 folds, 5 epochs
   - MAX_LEN = 224
   
2. **Phase 1 : Transformer Multi-Seed CV**
   - Entraîne `twitter-xlm-roberta-base` sur texte enrichi
   - StratifiedGroupKFold avec pseudo-user
   - Moyenne des 4 seeds
   
3. **Phase 2 : XGBoost GPU sur features**
   - 2000 estimators, early stopping
   - Sur les 45+ features de `feature_engineering.ipynb`
   
4. **Phase 3 : Meta-Ensemble**
   - Stack : [transformer_oof, xgb_oof]
   - Méta-modèle : LogisticRegression
   - Optimisation seuil sur meta_oof
   
5. **Phase 4 : Pseudo-labeling (optionnel)**
   - Sélectionne test samples avec proba > 0.85 ou < 0.15
   - Réentraîne transformer avec pseudo-labels
   - 2 epochs avec LR réduit

#### 🎯 Performance attendue
- **86-88% accuracy** (meilleur score du projet)

---

### 9️⃣ `notebooks/transformer_ultimate_v2.ipynb`

#### 📝 Description
**Version optimisée/expérimentale** de transformer_ultimate. (Tu as dit qu'il tourne actuellement)

#### 🔧 Différences probables vs v1
- Hyperparamètres ajustés
- Possiblement plus de seeds ou epochs
- Optimisations mémoire

---

### 🔟 `notebooks/train_model.ipynb` (NN Multi-Branch Simple)

#### 📝 Description
Réseau de neurones **simple à 3 branches** (tweet, user, meta) pour classification.

#### 📥 Entrées
- `data/embeddings/X_train_text_only_embeddings.npy`
- `data/embeddings/X_train_desc_only_embeddings.npy`
- `data/features/X_train_features.npy`
- `data/y_train.npy`

#### 📤 Sorties
- `models/influencer_model.pt`
- `submission/submission_NN.csv`

#### 🔧 Architecture
```
Tweet Embeddings (768d) → Linear(256) → ReLU → Dropout
User Embeddings (768d)  → Linear(256) → ReLU → Dropout
Meta Features (~45d)    → Linear(32)  → ReLU
                                 ↓
                         Concat (544d)
                                 ↓
                    Linear(128) → ReLU → Linear(64) → ReLU → Linear(2)
```

#### 🎯 Performance attendue
- ~83-84% accuracy (baseline NN)

---

### 1️⃣1️⃣ `notebooks/train_model_improved.ipynb` (NN Avancé) ⭐

#### 📝 Description
Version **améliorée** du NN avec attention fusion, EMA, multi-sample dropout, et cross-validation.

#### 📥 Entrées
Mêmes que `train_model.ipynb`

#### 📤 Sorties
- Modèles sauvegardés
- Submissions

#### 🔧 Améliorations vs train_model.ipynb
| Aspect | train_model | train_model_improved |
|--------|-------------|----------------------|
| Fusion | Concat simple | Attention Fusion |
| Dropout | Standard | Multi-Sample Dropout (5 masks) |
| Activation | ReLU | GELU |
| Régularisation | Basique | Label smoothing, EMA, gradient clipping |
| Validation | Simple split | StratifiedKFold CV |
| Early stopping | Non | Oui |

#### 🎯 Performance attendue
- ~84-85% accuracy

---

## 📊 TABLEAU RÉCAPITULATIF

| Notebook | Type | Entrées | Sorties | GPU | Perf estimée |
|----------|------|---------|---------|-----|--------------|
| preprocessing.ipynb | Preprocessing | jsonl | embeddings 776d | ✅ | - |
| preprocessing_improved.ipynb | Preprocessing | jsonl | embeddings 2304d+ | ✅ | - |
| **feature_engineering.ipynb** | Features | jsonl | features 45d | ❌ | - |
| model.ipynb | XGBoost | npy | predictions | ✅ | 83-84% |
| model_with_features.ipynb | NN+XGB+LGBM | npy+csv | models | ✅ | 85-86% |
| ensemble_stacking.ipynb | Stacking | csv+npy | predictions | ✅ | 86-87% |
| transformer_finetune.ipynb | Transformer | jsonl | predictions | ✅ | 84-85% |
| **transformer_ultimate.ipynb** | Full Pipeline | jsonl+npy | models+preds | ✅ | **86-88%** |
| train_model.ipynb | NN simple | npy | model | ✅ | 83-84% |
| train_model_improved.ipynb | NN avancé | npy | model | ✅ | 84-85% |

---

## 🚀 WORKFLOW RECOMMANDÉ

### Pour le meilleur score Kaggle :

```bash
# 1. Générer les features structurées (obligatoire pour transformer_ultimate)
→ Exécuter feature_engineering.ipynb

# 2. Lancer le pipeline complet
→ Exécuter notebooks/transformer_ultimate.ipynb (ou v2)

# 3. Soumettre
kaggle competitions submit -c influencer-or-observer \
    -f submission/submission_meta_ultimate.csv \
    -m "Multi-seed TF + XGB meta-LR"
```

### Pour explorer / comprendre :

```bash
# Comprendre les données et features
→ feature_engineering.ipynb

# Comprendre les embeddings
→ preprocessing_improved.ipynb

# Modèle simple baseline
→ model.ipynb

# Progresser vers le stacking
→ ensemble_stacking.ipynb
```

---

## ⚠️ RAPPEL IMPORTANT

**Colonnes qui N'EXISTENT PAS** dans les données brutes :
- ❌ `user.followers_count`
- ❌ `user.friends_count`
- ❌ `user.verified`

**Utiliser à la place :**
- ✅ `user.listed_count` (meilleur proxy d'influence)
- ✅ `user.statuses_count`, `user.favourites_count`
- ✅ Ratios calculés (`listed_per_status`, `tweets_per_favourites`)
