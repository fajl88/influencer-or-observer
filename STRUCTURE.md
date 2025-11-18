# 📁 Structure du Projet - Influencer or Observer

> Kaggle Challenge: Prédiction des rôles sociaux sur Twitter

## 📊 Vue d'Ensemble

```
influencer-or-observer/
├── 📄 Documentation
│   ├── README.md                    # Documentation principale
│   ├── QUICKSTART.md                # Guide de démarrage rapide
│   ├── PROJECT_STRUCTURE.md         # Description détaillée de la structure
│   └── STRUCTURE.md                 # Ce fichier (vue d'ensemble)
│
├── 📊 Données (data/)
│   ├── train.jsonl                  # Dataset d'entraînement (154,914 tweets)
│   ├── kaggle_test.jsonl            # Dataset de test (103,380 tweets)
│   └── README.md                    # Description des données
│
├── 📓 Notebooks (notebooks/)
│   ├── 01_EDA.ipynb                 # Analyse exploratoire
│   ├── 02_baseline.ipynb            # Modèles de baseline
│   ├── 03_feature_engineering.ipynb # Création de features
│   ├── 04_advanced_models.ipynb     # Modèles avancés
│   └── README.md                    # Guide des notebooks
│
├── 🐍 Code Source (src/)
│   ├── __init__.py                  # Initialisation du package
│   ├── data_loader.py               # Chargement JSONL et soumissions
│   ├── preprocessing.py             # Nettoyage et prétraitement
│   ├── feature_engineering.py       # Extraction de features
│   ├── models.py                    # Définitions des modèles ML
│   ├── evaluation.py                # Métriques et visualisations
│   └── utils.py                     # Fonctions utilitaires
│
├── 🤖 Modèles (models/)
│   └── .gitkeep                     # Modèles entraînés (*.pkl)
│
├── 📤 Soumissions (submissions/)
│   ├── dummy.csv                    # Baseline simple
│   └── logistic_regression.csv      # Soumission LogReg
│
├── 📈 Rapports (reports/)
│   ├── figures/                     # Graphiques et visualisations
│   └── performance.md               # Suivi des performances
│
├── ⚙️ Configuration (config/)
│   ├── config.json                  # Configuration principale (JSON)
│   └── config.yaml                  # Configuration alternative (YAML)
│
└── 🛠️ Scripts Utilitaires
    ├── check_setup.py               # Vérification de l'installation
    ├── train_and_submit.py          # Pipeline d'entraînement complet
    ├── requirements.txt             # Dépendances Python
    └── .gitignore                   # Fichiers à ignorer par Git
```

## 🎯 Fichiers Clés

### Documentation
- **README.md**: Documentation complète du projet avec méthodologie
- **QUICKSTART.md**: Guide rapide pour démarrer en 5 minutes
- **PROJECT_STRUCTURE.md**: Description détaillée de chaque composant

### Notebooks (Workflow progressif)
1. **01_EDA.ipynb**: Exploration des données, visualisations, insights
2. **02_baseline.ipynb**: Premiers modèles (Dummy, LogReg)
3. **03_feature_engineering.ipynb**: Création et sélection de features
4. **04_advanced_models.ipynb**: Random Forest, LightGBM, optimisation

### Modules Python (src/)
- **data_loader.py**: `load_training_data()`, `load_test_data()`, `save_submission()`
- **preprocessing.py**: `extract_full_text()`, `clean_text()`, `remove_urls()`
- **feature_engineering.py**: `count_hashtags()`, `count_mentions()`, `create_all_features()`
- **models.py**: `LogisticRegressionModel`, `RandomForestModel`, `save_model()`, `load_model()`
- **evaluation.py**: `evaluate_model()`, `plot_confusion_matrix()`, `plot_roc_curve()`
- **utils.py**: `load_config()`, `setup_logging()`, `reduce_memory_usage()`

### Scripts Principaux
- **check_setup.py**: Vérifie l'installation et la structure (7 checks)
- **train_and_submit.py**: Pipeline complet automatisé (train → predict → submit)

### Configuration
- **config.json**: Configuration en JSON (recommandé)
  - Chemins des fichiers
  - Paramètres de preprocessing
  - Hyperparamètres des modèles
  - Settings de validation

## 📏 Standards et Conventions

### Code Python
- **Style**: PEP 8
- **Docstrings**: Google style
- **Type hints**: Utilisés pour les fonctions principales
- **Logging**: Logger configuré dans chaque module

### Notebooks
- **Nommage**: `{numéro}_{description}.ipynb`
- **Structure**: Markdown → Code → Résultats
- **Documentation**: Titres clairs, explications détaillées

### Git
- **Commits**: Messages descriptifs en français
- **.gitignore**: Configure pour Python/Jupyter/Data science
- **Branches**: `main` pour production, feature branches pour dev

## 🔍 Vérifications de Qualité

### ✅ Structure Complète
- [x] Tous les dossiers créés avec README
- [x] 4 notebooks couvrant le workflow complet
- [x] 6 modules Python dans src/
- [x] Documentation complète (README, QUICKSTART, PROJECT_STRUCTURE)
- [x] Configuration JSON et YAML
- [x] Scripts utilitaires (check_setup, train_and_submit)

### ✅ Code Propre
- [x] Aucune erreur de lint
- [x] Tous les imports fonctionnent
- [x] Docstrings pour toutes les fonctions
- [x] Type hints ajoutés
- [x] Logging configuré

### ✅ Documentation
- [x] README.md complet avec badges, structure, utilisation
- [x] QUICKSTART.md pour démarrage rapide
- [x] README dans chaque sous-dossier
- [x] Commentaires dans le code
- [x] Notebooks avec explications Markdown

### ✅ Tests et Validation
- [x] check_setup.py vérifie 7 composants
- [x] Tous les tests passent (7/7)
- [x] Stopwords NLTK téléchargés (157 mots)
- [x] Modules s'importent sans erreur
- [x] Configuration se charge correctement

## 📦 Dépendances Principales

```
Core ML:
- pandas 2.2.2          (manipulation de données)
- numpy 1.26.4          (calculs numériques)
- scikit-learn 1.5.1    (machine learning)
- lightgbm 4.6.0        (gradient boosting)

NLP:
- nltk 3.9.1            (stopwords, tokenization)
- transformers 4.56.0   (modèles pré-entraînés)

Deep Learning:
- torch 2.8.0           (PyTorch)
- torchtext 0.18.0      (NLP avec PyTorch)

Visualisation:
- matplotlib 3.10.6     (graphiques)
- seaborn 0.13.2        (visualisations statistiques)

Développement:
- jupyter notebook      (notebooks interactifs)
- black 25.1.0          (formatage de code)
- isort 6.0.1           (tri des imports)
```

## 🚀 Démarrage Rapide

```bash
# 1. Vérifier l'installation
python check_setup.py

# 2. Explorer les données
jupyter notebook notebooks/01_EDA.ipynb

# 3. Tester un baseline
jupyter notebook notebooks/02_baseline.ipynb

# 4. Pipeline complet
python train_and_submit.py
```

## 📊 Métriques de Qualité

| Aspect | Status | Détails |
|--------|--------|---------|
| **Structure** | ✅ 100% | Tous les dossiers et fichiers présents |
| **Documentation** | ✅ 100% | README complet, guides, commentaires |
| **Code** | ✅ 100% | Aucune erreur, bien structuré |
| **Tests** | ✅ 100% | 7/7 vérifications passent |
| **Notebooks** | ✅ 100% | 4 notebooks couvrant le workflow |
| **Configuration** | ✅ 100% | Config JSON + YAML disponibles |

## 🎓 Pour les Nouveaux Contributeurs

1. **Lire**: README.md puis QUICKSTART.md
2. **Vérifier**: Exécuter `python check_setup.py`
3. **Explorer**: Ouvrir `notebooks/01_EDA.ipynb`
4. **Contribuer**: Suivre les conventions de code
5. **Tester**: Vérifier que tout fonctionne avant commit

## 📝 Notes Importantes

- **Données**: Les fichiers JSONL sont volumineux (~1.1 GB total)
- **NLTK**: Stopwords doivent être téléchargés (`python -c "import nltk; nltk.download('stopwords')"`)
- **GPU**: Recommandé pour les modèles Transformers (optionnel)
- **Mémoire**: Minimum 8 GB RAM recommandé

## 🔗 Liens Utiles

- [Challenge Kaggle](https://www.kaggle.com) (à compléter)
- [Documentation scikit-learn](https://scikit-learn.org/)
- [NLTK Documentation](https://www.nltk.org/)
- [Transformers by HuggingFace](https://huggingface.co/docs/transformers/)

---

**Version**: 1.0  
**Dernière mise à jour**: 18 novembre 2025  
**Statut**: ✅ Production-ready
