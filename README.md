# 🎯 Influencer or Observer: Predicting Social Roles

> Challenge Kaggle : Prédiction du rôle social des utilisateurs Twitter (Influenceur vs Observateur)

## 📋 Description du Projet

Ce projet vise à prédire le rôle social d'utilisateurs Twitter en analysant uniquement leurs tweets. Deux rôles principaux sont identifiés :

- **Influenceurs (1)** : Utilisateurs avec un réseau asymétrique et une grande audience (ratio followers/following élevé)
- **Observateurs (0)** : Utilisateurs avec un réseau équilibré et des conversations réciproques

### 🎓 Problématique

Est-ce que la stratégie réseau d'un utilisateur est encodée dans ses posts ? Peut-on déterminer le rôle social d'un utilisateur uniquement à partir de ses tweets ?

## 📊 Données

### Training Set
- **38,560 utilisateurs**
- **154,914 tweets**
- **194 features**
- Target binaire : 1 = Influencer, 0 = Observer

### Test Set
- **25,890 utilisateurs**
- **103,380 tweets**
- Prédictions à soumettre sur Kaggle

### Features Principales
- `full_text` : Le texte complet du tweet
- `source` : La source/plateforme utilisée
- `user.statuses_count` : Nombre de tweets de l'utilisateur
- `user.location` : Localisation
- Et ~190 autres features extraites de l'API Twitter

## 🏗️ Structure du Projet

```
influencer-or-observer/
│
├── data/                          # Données brutes et fichiers JSONL
│   ├── train.jsonl               # Dataset d'entraînement (154,914 tweets)
│   ├── kaggle_test.jsonl         # Dataset de test (103,380 tweets)
│   └── README.md                 # Description des données
│
├── notebooks/                     # Notebooks Jupyter pour l'exploration et expérimentation
│   ├── 01_EDA.ipynb              # Analyse exploratoire des données
│   ├── 02_baseline.ipynb         # Modèles de baseline
│   ├── 03_feature_engineering.ipynb  # Création de features
│   └── 04_advanced_models.ipynb  # Modèles avancés
│
├── src/                          # Code source Python modulaire
│   ├── __init__.py
│   ├── data_loader.py            # Chargement et parsing des données
│   ├── preprocessing.py          # Nettoyage et prétraitement
│   ├── feature_engineering.py    # Extraction de features
│   ├── models.py                 # Définitions des modèles
│   ├── evaluation.py             # Métriques et évaluation
│   └── utils.py                  # Fonctions utilitaires
│
├── models/                       # Modèles entraînés (pickles, joblib)
│   └── .gitkeep
│
├── submissions/                  # Fichiers CSV de soumission Kaggle
│   ├── dummy_submission.csv
│   └── logistic_regression_submission.csv
│
├── reports/                      # Rapports et visualisations
│   ├── figures/                  # Graphiques générés
│   └── performance.md            # Résumé des performances
│
├── config/                       # Fichiers de configuration
│   └── config.yaml               # Hyperparamètres et configurations
│
├── requirements.txt              # Dépendances Python
├── .gitignore                    # Fichiers à ignorer
└── README.md                     # Ce fichier
```

## 🚀 Installation

### Prérequis
- Python 3.8+
- pip

### Installation des dépendances

```bash
# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Télécharger les stopwords NLTK (si nécessaire)
python -c "import nltk; nltk.download('stopwords')"
```

## 📈 Méthodologie

### 1. Exploration des Données (EDA)
- Distribution des classes (Influencers vs Observers)
- Analyse textuelle (longueur des tweets, vocabulaire)
- Features utilisateur (statuses_count, location)
- Visualisations

### 2. Preprocessing
- Extraction du texte complet (`full_text` ou `extended_tweet.full_text`)
- Nettoyage du texte (URLs, mentions, hashtags)
- Normalisation

### 3. Feature Engineering
- **TF-IDF** : Vectorisation du texte (unigrams + bigrams)
- **Features textuelles** : Longueur, nombre de hashtags, mentions, emojis
- **Features utilisateur** : statuses_count, source, location
- **Features linguistiques** : Sentiment, complexité

### 4. Modélisation
- **Baseline** : Dummy Classifier (most_frequent)
- **Logistic Regression** : Avec TF-IDF
- **Modèles avancés** : 
  - Random Forest
  - XGBoost / LightGBM
  - Neural Networks
  - Transformers (CamemBERT pour le français)

### 5. Évaluation
- **Métrique principale** : Accuracy
- **Validation** : 5-Fold Stratified Cross-Validation
- **Test final** : Prédictions sur kaggle_test.jsonl

## 🎯 Métrique d'Évaluation

**Accuracy** : Proportion de prédictions correctes

$$
\text{Accuracy} = \frac{\text{Nombre de prédictions correctes}}{\text{Nombre total de prédictions}}
$$

## 📤 Format de Soumission

Fichier CSV avec deux colonnes :
- `ID` : Index de la ligne (entier à partir de 0)
- `Prediction` : Classe prédite (0 = Observer, 1 = Influencer)

Exemple :
```csv
ID,Prediction
0,1
2,1
4,0
8,1
9,0
```

## 📊 Résultats

| Modèle | Accuracy (CV) | Accuracy (Kaggle) | Notes |
|--------|--------------|-------------------|-------|
| Dummy (Most Frequent) | ~50% | - | Baseline |
| Logistic Regression | ~XX% | - | TF-IDF + French stopwords |
| - | - | - | À compléter |

## 🛠️ Utilisation

### Notebook d'exploration
```bash
jupyter notebook notebooks/01_EDA.ipynb
```

### Entraîner un modèle baseline
```bash
jupyter notebook notebooks/02_baseline.ipynb
```

### Utiliser les modules Python
```python
from src.data_loader import load_training_data, load_test_data
from src.preprocessing import extract_full_text
from src.models import train_logistic_regression

# Charger les données
X_train, y_train = load_training_data('data/train.jsonl')
X_test = load_test_data('data/kaggle_test.jsonl')

# Entraîner un modèle
model = train_logistic_regression(X_train, y_train)

# Faire des prédictions
predictions = model.predict(X_test)
```

## 📝 TODO / Améliorations Futures

- [ ] Analyse approfondie des features textuelles
- [ ] Feature engineering avancé (sentiment analysis, NER)
- [ ] Hyperparameter tuning avec GridSearchCV/RandomizedSearchCV
- [ ] Ensembling de modèles
- [ ] Utilisation de modèles de langage pré-entraînés (CamemBERT)
- [ ] Analyse des erreurs de classification
- [ ] A/B testing de différentes stratégies de preprocessing

## 👥 Auteur

[Votre Nom]

## 📄 Licence

Ce projet est développé dans le cadre d'un challenge Kaggle.

## 🔗 Liens Utiles

- [Page du challenge Kaggle](URL_DU_CHALLENGE)
- [Documentation scikit-learn](https://scikit-learn.org/)
- [Documentation Transformers](https://huggingface.co/docs/transformers/)

---

**Note** : Ce projet utilise des données Twitter et respecte les conditions d'utilisation de l'API Twitter et de Kaggle.