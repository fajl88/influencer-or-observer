# 📜 Changelog

Tous les changements notables de ce projet seront documentés dans ce fichier.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/),
et ce projet adhère au [Semantic Versioning](https://semver.org/lang/fr/).

## [1.0.0] - 2025-11-18

### ✨ Ajouté
- Structure complète du projet professionnel
- Documentation exhaustive (README, QUICKSTART, PROJECT_STRUCTURE, STRUCTURE, CONTRIBUTING)
- 6 modules Python dans `src/`:
  - `data_loader.py`: Chargement des données JSONL
  - `preprocessing.py`: Nettoyage et extraction de texte
  - `feature_engineering.py`: 15+ fonctions d'extraction de features
  - `models.py`: 3 classes de modèles (Baseline, LogisticRegression, RandomForest)
  - `evaluation.py`: Métriques et visualisations
  - `utils.py`: Configuration, logging, optimisation mémoire
- 4 notebooks Jupyter complets:
  - `01_EDA.ipynb`: Analyse exploratoire des données
  - `02_baseline.ipynb`: Modèles de baseline
  - `03_feature_engineering.ipynb`: Création et sélection de features
  - `04_advanced_models.ipynb`: Modèles avancés et optimisation
- Scripts utilitaires:
  - `check_setup.py`: Vérification de l'installation (7 checks)
  - `train_and_submit.py`: Pipeline d'entraînement complet
- Configuration JSON et YAML avec tous les hyperparamètres
- Support complet NLTK avec stopwords français (157 mots)
- Intégration complète de scikit-learn, LightGBM, Transformers
- Système de logging configuré
- .gitignore complet pour Python/Jupyter/Data Science

### 🔧 Configuration
- Python 3.8+ (testé avec 3.12.11)
- 34 dépendances dans requirements.txt
- Support GPU pour PyTorch (optionnel)

### 📊 Features
- Extraction de texte complet (tweets normaux et tronqués)
- 15+ features textuelles (hashtags, mentions, emojis, URLs, etc.)
- TF-IDF avec paramètres configurables
- Validation croisée stratifiée
- Métriques complètes (accuracy, confusion matrix, ROC curve)
- Sauvegarde/chargement de modèles avec pickle

### 📚 Documentation
- README.md: 250+ lignes avec structure, méthodologie, utilisation
- QUICKSTART.md: Guide de démarrage en 5 minutes
- PROJECT_STRUCTURE.md: Description détaillée de chaque composant
- STRUCTURE.md: Vue d'ensemble avec checklist qualité
- CONTRIBUTING.md: Guide de contribution complet
- README dans chaque sous-dossier

### ✅ Tests
- Script check_setup.py avec 7 vérifications:
  - Version Python
  - Packages installés
  - Données NLTK
  - Structure des dossiers
  - Fichiers de données
  - Fichiers de configuration
  - Modules source
- Tous les tests passent (7/7)
- Aucune erreur de lint

### 🎯 Qualité
- Code formaté selon PEP 8
- Docstrings Google-style pour toutes les fonctions
- Type hints pour fonctions principales
- Logging configuré dans tous les modules
- Gestion d'erreurs appropriée
- Code testé et validé

### 📈 Performance
- Dummy Classifier (baseline): ~50% accuracy
- Logistic Regression + TF-IDF: À compléter
- Random Forest + TF-IDF: À compléter
- LightGBM: À compléter

## [Unreleased]

### 🔮 À Venir
- [ ] Fine-tuning de CamemBERT pour le français
- [ ] Hyperparameter tuning avec GridSearchCV
- [ ] Ensemble methods (stacking, voting)
- [ ] Features de sentiment analysis
- [ ] Named Entity Recognition (NER)
- [ ] Analyse temporelle des tweets
- [ ] Dashboard interactif avec Streamlit
- [ ] Tests unitaires automatisés
- [ ] CI/CD avec GitHub Actions
- [ ] Docker containerization

### 💡 Idées
- Analyse des erreurs de classification
- Features basées sur les emojis spécifiques
- Détection de langues multiples
- Analyse du réseau social (si données disponibles)
- Cross-validation temporelle
- Data augmentation pour équilibrer les classes

## Notes de Version

### Format de Versioning
Le projet utilise [Semantic Versioning](https://semver.org/):
- **MAJOR**: Changements incompatibles de l'API
- **MINOR**: Nouvelles fonctionnalités (rétrocompatibles)
- **PATCH**: Corrections de bugs (rétrocompatibles)

### Types de Changements
- **Ajouté** (`Added`): Nouvelles fonctionnalités
- **Modifié** (`Changed`): Changements dans les fonctionnalités existantes
- **Déprécié** (`Deprecated`): Fonctionnalités qui seront retirées
- **Retiré** (`Removed`): Fonctionnalités retirées
- **Corrigé** (`Fixed`): Corrections de bugs
- **Sécurité** (`Security`): Vulnérabilités corrigées

---

**Légende**:
- ✨ Nouveau
- 🔧 Configuration
- 🐛 Correction
- 📚 Documentation
- ⚡ Performance
- 🎨 Style
- ♻️ Refactoring
- 🔒 Sécurité
