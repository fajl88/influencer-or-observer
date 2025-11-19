# Structure Détaillée du Projet

Ce document décrit la structure complète du projet Influencer-or-Observer.

## 📁 Vue d'Ensemble

```
influencer-or-observer/
│
├── 📄 README.md                    # Documentation principale
├── 📄 QUICKSTART.md                # Guide de démarrage rapide
├── 📄 PROJECT_STRUCTURE.md         # Ce fichier
├── 📄 requirements.txt             # Dépendances Python
├── 📄 .gitignore                   # Fichiers à ignorer par Git
├── 🐍 train_and_submit.py          # Script d'entraînement complet
│
├── 📂 data/                        # Données du projet
│   ├── 📄 README.md                # Description des données
│   ├── 📊 train.jsonl              # Dataset d'entraînement (154,914 tweets)
│   └── 📊 kaggle_test.jsonl        # Dataset de test (103,380 tweets)
│
├── 📂 notebooks/                   # Notebooks Jupyter
│   ├── 📄 README.md                # Organisation des notebooks
│   ├── 📓 01_EDA.ipynb             # Analyse exploratoire (à créer)
│   ├── 📓 02_baseline.ipynb        # Modèles de baseline
│   ├── 📓 03_feature_engineering.ipynb  # Feature engineering (à créer)
│   └── 📓 04_advanced_models.ipynb # Modèles avancés (à créer)
│
├── 📂 src/                         # Code source Python
│   ├── 🐍 __init__.py              # Initialisation du package
│   ├── 🐍 data_loader.py           # Chargement et parsing des données
│   ├── 🐍 preprocessing.py         # Nettoyage et prétraitement du texte
│   ├── 🐍 feature_engineering.py   # Extraction de features
│   ├── 🐍 models.py                # Définitions des modèles ML
│   ├── 🐍 evaluation.py            # Métriques et évaluation
│   └── 🐍 utils.py                 # Fonctions utilitaires
│
├── 📂 config/                      # Configuration
│   └── ⚙️ config.yaml              # Hyperparamètres et paramètres
│
├── 📂 models/                      # Modèles entraînés
│   └── 📄 .gitkeep                 # Documentation du dossier
│
├── 📂 submissions/                 # Fichiers de soumission Kaggle
│   ├── 📊 dummy.csv                # Soumission baseline (Dummy)
│   └── 📊 logistic_regression.csv  # Soumission Logistic Regression
│
└── 📂 reports/                     # Rapports et visualisations
    ├── 📄 performance.md           # Résumé des performances
    └── 📂 figures/                 # Graphiques et visualisations
        └── 📄 .gitkeep             # Documentation du dossier
```

## 📦 Modules Python (src/)

### data_loader.py
**Fonctions principales:**
- `load_jsonl()`: Charge un fichier JSONL
- `load_training_data()`: Charge et sépare X_train, y_train
- `load_test_data()`: Charge les données de test
- `save_submission()`: Crée un fichier de soumission Kaggle
- `get_dataset_info()`: Statistiques sur le dataset

**Usage:**
```python
from src.data_loader import load_training_data
X_train, y_train = load_training_data('data/train.jsonl')
```

### preprocessing.py
**Fonctions principales:**
- `extract_full_text()`: Extrait le texte complet d'un tweet
- `clean_text()`: Nettoie le texte (URLs, mentions, etc.)
- `remove_urls()`, `remove_mentions()`, `remove_hashtags()`
- `preprocess_dataframe()`: Prétraite un DataFrame complet
- `get_text_statistics()`: Statistiques textuelles

**Usage:**
```python
from src.preprocessing import preprocess_dataframe
X_train = preprocess_dataframe(X_train)
```

### feature_engineering.py
**Fonctions principales:**
- `count_hashtags()`, `count_mentions()`, `count_urls()`, `count_emojis()`
- `get_text_length()`, `get_word_count()`, `get_avg_word_length()`
- `is_retweet()`, `is_quote()`, `is_reply()`
- `extract_source_type()`: Type de plateforme utilisée
- `create_text_features()`: Crée features textuelles
- `create_all_features()`: Crée toutes les features

**Usage:**
```python
from src.feature_engineering import create_all_features
X_train = create_all_features(X_train)
```

### models.py
**Classes principales:**
- `BaselineModel`: Dummy Classifier
- `LogisticRegressionModel`: Logistic Regression + TF-IDF
- `RandomForestModel`: Random Forest + TF-IDF
- `save_model()`, `load_model()`: Sauvegarde/chargement

**Usage:**
```python
from src.models import LogisticRegressionModel
model = LogisticRegressionModel()
model.fit(X_train['full_text'], y_train)
predictions = model.predict(X_test['full_text'])
```

### evaluation.py
**Fonctions principales:**
- `evaluate_model()`: Calcule accuracy, precision, recall, F1
- `print_classification_report()`: Rapport détaillé
- `plot_confusion_matrix()`: Matrice de confusion
- `plot_roc_curve()`: Courbe ROC
- `compare_models()`: Compare plusieurs modèles
- `plot_feature_importance()`: Importance des features

**Usage:**
```python
from src.evaluation import evaluate_model
metrics = evaluate_model(y_true, y_pred, y_pred_proba)
```

### utils.py
**Fonctions principales:**
- `setup_logging()`: Configure le logging
- `load_config()`: Charge config.yaml
- `save_json()`, `load_json()`: Manipulation JSON
- `get_class_distribution()`: Distribution des classes
- `reduce_memory_usage()`: Optimise la mémoire
- `print_section()`: Affichage formaté

**Usage:**
```python
from src.utils import load_config, setup_logging
config = load_config('config/config.yaml')
setup_logging(level='INFO')
```

## ⚙️ Configuration (config/config.yaml)

Sections principales:
- **paths**: Chemins des fichiers
- **preprocessing**: Paramètres de nettoyage
- **tfidf**: Paramètres TF-IDF (max_features, ngrams, stopwords)
- **logistic_regression**: Hyperparamètres
- **random_forest**, **xgboost**, **lightgbm**: Hyperparamètres
- **validation**: Cross-validation (folds, stratified)
- **features**: Features à utiliser

## 📓 Notebooks

### 01_EDA.ipynb (À créer)
- Distribution des classes
- Analyse du texte (longueur, vocabulaire)
- Visualisations
- Features utilisateur
- Word clouds

### 02_baseline.ipynb (Existant)
- Dummy Classifier
- Logistic Regression + TF-IDF
- Cross-validation
- Soumissions de base

### 03_feature_engineering.ipynb (À créer)
- Création de features textuelles
- Features utilisateur
- Analyse d'importance
- Sélection de features

### 04_advanced_models.ipynb (À créer)
- Random Forest
- XGBoost, LightGBM
- Neural Networks
- Fine-tuning CamemBERT
- Ensemble methods

## 🚀 Workflow Typique

1. **Exploration**: `notebooks/01_EDA.ipynb`
2. **Baseline**: `notebooks/02_baseline.ipynb`
3. **Feature Engineering**: `notebooks/03_feature_engineering.ipynb`
4. **Modèles Avancés**: `notebooks/04_advanced_models.ipynb`
5. **Production**: `train_and_submit.py`

## 📝 Conventions

### Nommage des fichiers
- **Notebooks**: `{numéro}_{description}.ipynb`
- **Modèles**: `{model_name}_{date}_{score}.joblib`
- **Soumissions**: `{model_name}_{score}.csv`
- **Figures**: `{type}_{description}_{date}.png`

### Code
- Docstrings pour toutes les fonctions
- Type hints quand possible
- Logging plutôt que print dans les modules
- Configuration via config.yaml

## 🔍 Fichiers Importants

| Fichier | Description | Quand l'utiliser |
|---------|-------------|------------------|
| `README.md` | Documentation complète | Vue d'ensemble du projet |
| `QUICKSTART.md` | Guide de démarrage | Premier contact avec le projet |
| `train_and_submit.py` | Script complet | Entraînement automatisé |
| `config/config.yaml` | Configuration | Modifier les hyperparamètres |
| `requirements.txt` | Dépendances | Installation |

## 📊 Formats de Données

### train.jsonl
```json
{
  "text": "...",
  "extended_tweet": {"full_text": "..."},
  "user": {"statuses_count": 1234, ...},
  "label": 0 ou 1,
  "challenge_id": 123,
  ...
}
```

### Soumission Kaggle (CSV)
```csv
ID,Prediction
0,1
1,0
2,1
...
```

## 🛠️ Dépendances Principales

- **pandas**: Manipulation de données
- **numpy**: Calculs numériques
- **scikit-learn**: Modèles ML classiques
- **nltk**: NLP (stopwords)
- **matplotlib**, **seaborn**: Visualisations
- **transformers**: Modèles de langage (CamemBERT)
- **lightgbm**, **xgboost**: Gradient boosting
- **torch**: Deep learning

## 📈 Progrès

- [x] Structure du projet créée
- [x] Modules Python implémentés
- [x] Configuration YAML
- [x] Documentation complète
- [x] Notebook baseline
- [ ] Notebook EDA
- [ ] Notebook feature engineering
- [ ] Notebook modèles avancés
- [ ] Fine-tuning CamemBERT
- [ ] Ensemble de modèles

---

**Dernière mise à jour**: 18 Novembre 2025
