# 🐦 Influencer or Observer

> **Kaggle Competition** — Classification binaire de tweets francophones : *Influencer* vs *Observer*

## 📋 Table des matières

- [Vue d'ensemble](#-vue-densemble)
- [Installation](#-installation)
- [Structure du projet](#-structure-du-projet)
- [Pipeline recommandé](#-pipeline-recommandé)
- [Description des notebooks](#-description-des-notebooks)
- [Données](#-données)
- [Techniques ML utilisées](#-techniques-ml-utilisées)
- [Résultats](#-résultats)

---

## 🎯 Vue d'ensemble

Ce projet implémente un pipeline complet de classification de tweets francophones pour distinguer les **influenceurs** des **observateurs**. Il combine :

- **Embeddings CamemBERT** (multi-couches) pour capturer la sémantique
- **Features engineered** (textuelles, structurelles, utilisateur)
- **Modèles ensemblistes** (LightGBM, XGBoost, Neural Networks)
- **Stacking** pour combiner les prédictions

---

## 🚀 Installation

```bash
# 1. Cloner le repository
git clone <repo-url>
cd influencer-or-observer

# 2. Créer l'environnement virtuel
python -m venv kaggle-env
source kaggle-env/bin/activate  # Linux/Mac
# kaggle-env\Scripts\activate   # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. (Optionnel) Pour l'optimisation Optuna
pip install optuna
```

### Dépendances principales

| Package | Usage |
|---------|-------|
| `transformers` | CamemBERT embeddings |
| `torch` | Neural networks |
| `lightgbm`, `xgboost` | Gradient boosting |
| `scikit-learn` | Preprocessing, métriques |
| `pandas`, `numpy` | Manipulation de données |

---

## 📁 Structure du projet

```
influencer-or-observer/
│
├── 📊 NOTEBOOKS PRINCIPAUX
│   ├── preprocessing_improved.ipynb   # Preprocessing professionnel
│   ├── feature_engineering.ipynb      # Extraction de features
│   ├── model_with_features.ipynb      # Modèle NN + features
│   ├── ensemble_stacking.ipynb        # Ensemble & stacking
│   └── model.ipynb                    # XGBoost baseline
│
├── 🚀 NOTEBOOKS AVANCÉS
│   └── notebooks/transformer_ultimate_v2.ipynb  # Twitter XLM-R large + TF‑IDF + XGB + stacking calibré
│
├── 📁 notebooks/
│   ├── train_model.ipynb              # Training NN original
│   └── train_model_improved.ipynb     # Training NN amélioré (CV, EMA)
│
├── 📁 data/
│   ├── train.jsonl                    # Données d'entraînement
│   ├── kaggle_test.jsonl              # Données de test
│   ├── y_train.npy                    # Labels
│   ├── X_train_processed*.npy         # Features processées
│   ├── embeddings/                    # Embeddings CamemBERT
│   └── features/                      # Features structurées
│
├── 📁 submission/                     # Fichiers CSV de soumission Kaggle
│   ├── submission_features.csv        # LightGBM (features only) - 83.84%
│   ├── submission_model_with_features.csv  # NN + embeddings + features
│   ├── ensemble_weighted.csv          # Weighted ensemble
│   ├── ensemble_stacking.csv          # Stacking meta-model
│   └── submission_XGBoost.csv         # XGBoost baseline
├── 📁 labs/                           # Labs du cours (référence)
├── requirements.txt
├── TODO.md                            # Stratégie détaillée
└── README.md
```

---

## 🔄 Pipeline recommandé

### Option A : Pipeline rapide (< 30 min)

```
1️⃣ preprocessing_improved.ipynb (cellules 1-8 seulement, sans régénérer embeddings)
    ↓
2️⃣ model.ipynb (XGBoost avec features existantes)
    ↓
3️⃣ Soumission Kaggle
```

### Option B : Pipeline complet (meilleure performance)

```
1️⃣ preprocessing_improved.ipynb
    • Extraire features textuelles et structurées
    • (Optionnel) Régénérer embeddings multi-couches [GPU recommandé]
    ↓
2️⃣ feature_engineering.ipynb
    • Features avancées (sentiment, NER, patterns)
    ↓
3️⃣ model_with_features.ipynb
    • Entraîner Neural Network avec attention fusion
    ↓
4️⃣ notebooks/train_model_improved.ipynb
    • Cross-validation 5-folds
    • Early stopping, EMA
    ↓
5️⃣ ensemble_stacking.ipynb
    • Combiner LightGBM + XGBoost + NN
    • Optimisation Optuna des poids
    ↓
6️⃣ Soumission Kaggle 🏆
```

### Option C : Pipeline leaderboard (plus robuste CV/LB)
```
1️⃣ notebooks/transformer_ultimate_v2.ipynb
    • Twitter XLM-R large (multi-seed) avec label smoothing + LLRD
    • TF‑IDF char/word + Logistic Regression (forte généralisation)
    • XGBoost sur features engineered
    • Stacking calibré pour réduire l'écart CV/LB
2️⃣ Soumettre `submission_meta_v2.csv` ou `submission_tf_v2_thrXX.csv`
```

---

## 📓 Description des notebooks

### 1. `preprocessing_improved.ipynb` ⭐

**Rôle** : Preprocessing professionnel des données brutes

| Étape | Description |
|-------|-------------|
| Configuration | Dataclass centralisée, logging |
| Chargement | Parse JSONL, normalisation |
| Text Features | 20+ features (hashtags, mentions, emojis, CTA...) |
| Structured Features | Colonnes numériques et catégorielles |
| Embeddings | CamemBERT multi-couches (layers -1,-2,-3,-4) |
| Validation | Vérification NaN/Inf |

```python
# Pour régénérer les embeddings (⚠️ lent, GPU recommandé)
REGENERATE_EMBEDDINGS = True
```

---

### 2. `feature_engineering.ipynb`

**Rôle** : Extraction de features discriminantes

- **Features textuelles** : longueur, mots, ponctuation
- **Features structurelles** : retweet count, favorite count
- **Features utilisateur** : followers ratio, listed count
- **Features NLP** : sentiment analysis, NER entities

**Output** : `data/train_features.csv`, `data/test_features.csv`

---

### 3. `model_with_features.ipynb` ⭐

**Rôle** : Neural network combinant embeddings + features

**Architecture** :
```
Tweet Embeddings ──┐
                   │
User Embeddings ───┼── Attention Fusion ── Classifier
                   │
Engineered Features┘
```

**Techniques** :
- Multi-Sample Dropout
- Label Smoothing
- Gradient Clipping
- Out-of-Fold predictions

---

### 4. `notebooks/train_model_improved.ipynb` ⭐

**Rôle** : Training NN professionnel avec CV

| Feature | Description |
|---------|-------------|
| Cross-validation | 5-fold stratifié |
| Early Stopping | Patience = 7, monitoring F1 |
| EMA | Exponential Moving Average des poids |
| AMP | Mixed precision training |
| Scheduler | OneCycleLR |

---

### 5. `ensemble_stacking.ipynb` ⭐

**Rôle** : Ensemble et méta-modèle

**Modèles de base** :
- LightGBM
- XGBoost  
- Neural Network
- Logistic Regression

**Méthodes d'ensemble** :
- Voting (soft)
- Stacking (méta-modèle)
- Optimisation Optuna des poids

---

### 6. `model.ipynb` (Baseline)

**Rôle** : XGBoost baseline avec GridSearchCV

- Pipeline sklearn standard
- Validation split 85/15
- Génère `submission/submission.csv`

---

## 📊 Données

| Fichier | Description | Samples |
|---------|-------------|---------|
| `train.jsonl` | Tweets labellisés | ~155k |
| `kaggle_test.jsonl` | Tweets à prédire | ~103k |

### Features générées

| Type | Nombre | Exemples |
|------|--------|----------|
| Textuelles | 25+ | hashtag_count, emoji_count, has_url, has_call_to_action |
| Structurelles | 15+ | retweet_count, total_engagement, source_device |
| Utilisateur | 12+ | user_listed_count, user_has_url, user_has_banner |
| Dérivées | 8+ | listed_per_status, tweets_per_favourites, log_user_listed |
| Embeddings | 768-3072 | CamemBERT (1-4 couches) |

> ⚠️ **Note**: Les colonnes `user.followers_count`, `user.friends_count`, `user.verified` **n'existent pas** dans les données. Nous utilisons les features réellement disponibles.

---

## 🧠 Techniques ML utilisées

### Inspirées des Labs du cours

| Lab | Technique | Notebook |
|-----|-----------|----------|
| **Lab 4** | Dropout, BatchNorm, Weight Decay | `train_model_improved.ipynb` |
| **Lab 5** | Attention, Transformers | `preprocessing_improved.ipynb` |
| **Lab 6** | Feature Engineering | `feature_engineering.ipynb` |
| **Lab 8** | Optimisation, Scheduling | `model_with_features.ipynb` |

### Meilleures pratiques

- ✅ **Embeddings multi-couches** : Concat layers -1,-2,-3,-4
- ✅ **Attention pooling** : Meilleur que CLS token
- ✅ **Label smoothing** : Meilleure généralisation
- ✅ **EMA** : Stabilité des poids
- ✅ **Cross-validation** : Estimation fiable du score
- ✅ **Stacking** : Combine forces des modèles

---

## 📈 Résultats

| Approche | Accuracy (CV) | Notebook |
|----------|---------------|----------|
| LightGBM (features only) | **83.84%** ⭐ | `feature_engineering.ipynb` |
| XGBoost + Embeddings | ~82-84% | `model.ipynb` |
| NN + Features | ~84-85% | `model_with_features.ipynb` |
| Ensemble Stacking | ~85%+ | `ensemble_stacking.ipynb` |

### Top Features Découvertes

| Feature | Importance LightGBM | Corrélation |
|---------|---------------------|-------------|
| `user_description_length` | 736 ⭐ | - |
| `tweets_per_favourites` | 699 ⭐ | - |
| `user_favourites_count` | 646 ⭐ | - |
| `log_user_listed` | - | 0.606 ⭐ |
| `user_has_url` | 130 | 0.420 |

---

## ⚡ Quick Start

```bash
# 1. Activer l'environnement
source kaggle-env/bin/activate

# 2. Lancer Jupyter
jupyter notebook

# 3. Exécuter dans l'ordre :
#    - preprocessing_improved.ipynb
#    - model_with_features.ipynb  
#    - ensemble_stacking.ipynb

# 4. Soumettre sur Kaggle
#    Fichiers disponibles dans submission/:
#    - submission_features.csv         (LightGBM features only, 83.84%)
#    - submission_model_with_features.csv  (NN + embeddings + features)
#    - ensemble_weighted.csv           (Weighted ensemble)
#    - ensemble_stacking.csv           (Stacking meta-model)
```

---

## 📝 Notes importantes

> ⚠️ **Embeddings** : La génération des embeddings CamemBERT prend ~2-3h sur GPU. Utilisez les embeddings pré-calculés si disponibles.

> 💡 **GPU** : Recommandé pour `preprocessing_improved.ipynb` et `train_model_improved.ipynb`

> 🔧 **Paths** : Vérifiez `DATA_DIR` au début de chaque notebook si nécessaire

---

## 👥 Auteurs

Projet réalisé dans le cadre du cours de Machine Learning.

---

## 📄 License

MIT License
