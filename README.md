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
│   ├── 04_advanced_models.ipynb  # 🚀 Modèles avancés (PRINCIPAL)
│   ├── ADVANCED_MODELS_GUIDE.md  # 📖 Guide complet des modèles
│   └── README.md
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
├── docs/                         # 📚 Documentation détaillée
│   ├── README.md                 # Index de la documentation
│   ├── IMPROVEMENTS_SUMMARY.md   # Résumé des améliorations
│   └── PIPELINE_VISUAL.md        # Visualisation du pipeline
│
├── config/                       # Fichiers de configuration
│   └── config.yaml               # Hyperparamètres et configurations
│
├── requirements.txt              # Dépendances Python
├── QUICKSTART.md                 # ⚡ Guide de démarrage rapide
├── generate_submissions.py       # Script pour générer les soumissions
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

### 3. Feature Engineering 🔧
- **TF-IDF Avancé** : Vectorisation avec trigrammes (1,2,3), 100K features
- **Features textuelles extraites** :
  - Longueur du texte, nombre de mots
  - Comptage : hashtags, mentions, URLs, emojis
  - Style : ratio majuscules, ponctuation (!, ?)
  - Nombres mentionnés
- **Features utilisateur** : 194 features numériques natives du dataset
- **Agrégation statistique** : mean, std, max, sum par utilisateur
- **Normalisation** : StandardScaler pour features numériques

### 4. Modélisation 🤖
#### Modèles Classiques Optimisés
- **Logistic Regression** : TF-IDF trigrammes + features numériques, GridSearch sur C
- **SVM** : LinearSVC avec class_weight='balanced'
- **Random Forest** : 200 estimateurs avec features combinées
- **LightGBM** : 500 estimateurs, learning_rate=0.05

#### Transformers
- **CamemBERT** : Fine-tuning optimisé pour Mac M4
  - Mode échantillon (15%) pour tests rapides
  - Mode complet pour production
  - Batch size adapté à la mémoire unifiée

#### Ensembles ⭐
- **Vote Majoritaire** : Consensus de 4 modèles classiques
- **Moyenne de Probabilités** : Combinaison pondérée
- **Recommandé pour soumission finale**

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

## 📊 Résultats et Performances

### Modèles Implémentés

| Modèle | Accuracy (CV) | Features | Notes |
|--------|--------------|----------|-------|
| Logistic Regression | ~XX% | TF-IDF trigrammes + 194 numériques | GridSearch optimisé |
| SVM | ~XX% | TF-IDF trigrammes | Class balanced |
| Random Forest | ~XX% | TF-IDF + numériques | 200 estimateurs |
| LightGBM | ~XX% | TF-IDF + numériques | 500 estimateurs |
| **Ensemble Vote** ⭐ | ~XX% | Tous classiques | Recommandé |
| **Ensemble Mean** ⭐ | ~XX% | Tous classiques | Recommandé |
| CamemBERT | ~XX% | Transformers FR | Fine-tuned |

> **Note**: Exécutez `04_advanced_models.ipynb` pour obtenir les scores exacts

### Fichiers de Soumission Générés

Les soumissions sont créées avec **agrégation par utilisateur** (essentiel pour Kaggle) :

1. `ensemble_mean_submission.csv` ⭐⭐⭐ (Priorité 1)
2. `ensemble_vote_submission.csv` ⭐⭐⭐ (Priorité 1)
3. `logistic_regression_advanced_submission.csv` ⭐⭐
4. `svm_advanced_submission.csv` ⭐⭐
5. `random_forest_advanced_submission.csv` ⭐⭐
6. `lightgbm_advanced_submission.csv` ⭐⭐
7. `camembert_submission.csv` ⭐⭐ (si entraîné en mode complet)

## 🛠️ Utilisation

### 🚀 Méthode Rapide (Recommandée)

```bash
# 1. Ouvrir le notebook principal
jupyter notebook notebooks/04_advanced_models.ipynb

# 2. Exécuter toutes les cellules (menu Cell > Run All)
#    Temps estimé: 15-30 min sur Mac M4

# 3. Les soumissions sont générées automatiquement dans submissions/
```

### 📖 Guide Détaillé

Consultez le guide complet : [`notebooks/ADVANCED_MODELS_GUIDE.md`](notebooks/ADVANCED_MODELS_GUIDE.md)

### ⚡ Mode Échantillon (Test Rapide)

Pour tester le pipeline rapidement (~10 min) :

```python
# Dans la cellule CamemBERT du notebook
USE_SAMPLE = True   # Mode test avec 15% des données
SAMPLE_SIZE = 0.15
```

### 🔥 Mode Production (Soumission Finale)

Pour les meilleurs résultats (~30-40 min) :

```python
# Dans la cellule CamemBERT du notebook
USE_SAMPLE = False  # Utilise toutes les données
```

### 🐍 Script Python (Alternative)

```bash
# Générer toutes les soumissions
python generate_submissions.py --models all

# Générer uniquement les ensembles
python generate_submissions.py --models ensemble

# Générer uniquement CamemBERT
python generate_submissions.py --models camembert
```

## 📝 Améliorations Implémentées ✅

- ✅ **Feature Engineering Avancé**
  - Extraction de features textuelles (hashtags, mentions, emojis, majuscules)
  - Utilisation des 194 features numériques natives
  - Agrégation statistique par utilisateur (mean, std, max, sum)
  - Normalisation StandardScaler

- ✅ **TF-IDF Optimisé**
  - Trigrammes (1,2,3) pour capturer plus de contexte
  - 100K features max au lieu de 1K
  - min_df pour filtrer mots rares
  - Sublinear TF scaling

- ✅ **Modèles Classiques Renforcés**
  - Logistic Regression avec GridSearch sur C
  - SVM avec class_weight='balanced'
  - Random Forest et LightGBM avec features combinées

- ✅ **Transformers pour le Français**
  - CamemBERT fine-tuned
  - Optimisé pour Mac M4 (MPS backend)
  - Mode échantillon pour tests rapides

- ✅ **Ensembles de Modèles**
  - Vote majoritaire
  - Moyenne de probabilités
  - Combine tous les modèles classiques

- ✅ **Agrégation par Utilisateur**
  - Essentielle pour la métrique Kaggle
  - Moyenne des probas par challenge_id
  - Vote majoritaire des tweets d'un même utilisateur

- ✅ **Optimisations Mac M4**
  - Utilisation du GPU Apple Silicon (MPS)
  - Parallélisation multi-core (10 cœurs)
  - Batch size optimisé pour mémoire unifiée

### 🎯 Prochaines Améliorations Possibles

- [ ] Data augmentation (back-translation)
- [ ] Stacking avec meta-modèle
- [ ] XLM-RoBERTa pour comparaison
- [ ] RandomizedSearchCV pour LightGBM
- [ ] Analyse approfondie des erreurs
- [ ] Features sociales (graphe d'influence)
- [ ] Pseudo-labeling du test set

## 💻 Optimisé pour Mac M4

Ce projet tire parti des capacités exceptionnelles du **Mac M4** :

### Spécifications M4 Max
- 🧠 **Neural Engine** 16-core
- 🎮 **GPU** jusqu'à 40-core
- 💾 **Memory bandwidth** jusqu'à 546 GB/s
- ⚡ **Architecture unifiée** (pas de transfert CPU↔GPU)

### Optimisations Appliquées
```python
torch.set_num_threads(10)              # Utiliser les 10 cœurs
device = torch.device("mps")           # GPU Apple Silicon
per_device_train_batch_size=32         # Gros batch possible
dataloader_num_workers=4               # Parallélisme I/O
```

### Temps d'Exécution Estimés
- Modèles classiques (tous) : **5-15 min**
- CamemBERT échantillon (15%) : **5-10 min**
- CamemBERT complet : **20-40 min** (3 epochs)
- **Total pipeline complet : ~30-45 min**

## 📚 Références Scientifiques

Les implémentations sont basées sur des recherches récentes :

1. **TF-IDF + LogReg** : Papier 2024 montrant 90% accuracy sur tweets
2. **SVM Trigrammes** : Études comparatives sur classification de texte
3. **CamemBERT** : Modèle RoBERTa français battant mBERT et XLM-R
4. **Class Balancing** : Pratique standard pour datasets déséquilibrés
5. **Ensemble Methods** : Approche classique des top Kagglers

Voir [`notebooks/ADVANCED_MODELS_GUIDE.md`](notebooks/ADVANCED_MODELS_GUIDE.md) pour les références complètes.

## � Documentation Complète

### 🚀 Pour Démarrer
- **[QUICKSTART.md](QUICKSTART.md)** - Guide de démarrage rapide (5 min)
- **[notebooks/04_advanced_models.ipynb](notebooks/04_advanced_models.ipynb)** - Notebook principal à exécuter

### 📖 Documentation Technique
- **[docs/](docs/)** - Documentation détaillée
  - [IMPROVEMENTS_SUMMARY.md](docs/IMPROVEMENTS_SUMMARY.md) - Résumé de toutes les améliorations
  - [PIPELINE_VISUAL.md](docs/PIPELINE_VISUAL.md) - Visualisation du pipeline ML
- **[notebooks/ADVANCED_MODELS_GUIDE.md](notebooks/ADVANCED_MODELS_GUIDE.md)** - Guide complet des modèles (200+ lignes)

### 🔧 Autres Documents
- [CONTRIBUTING.md](CONTRIBUTING.md) - Guide de contribution
- [CHANGELOG.md](CHANGELOG.md) - Historique des changements
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Structure détaillée du projet

## �👥 Auteur

[Votre Nom]

## 📄 Licence

Ce projet est développé dans le cadre d'un challenge Kaggle.

## 🔗 Liens Utiles

- [Page du challenge Kaggle](URL_DU_CHALLENGE)
- [Documentation scikit-learn](https://scikit-learn.org/)
- [Documentation Transformers](https://huggingface.co/docs/transformers/)

---

**Note** : Ce projet utilise des données Twitter et respecte les conditions d'utilisation de l'API Twitter et de Kaggle.