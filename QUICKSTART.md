# Guide de Démarrage Rapide

Ce guide vous aide à démarrer rapidement avec le projet Influencer-or-Observer.

## 🚀 Installation Rapide

```bash
# 1. Cloner le repository (si applicable)
git clone <url>
cd influencer-or-observer

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur macOS/Linux
# venv\Scripts\activate   # Sur Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Télécharger les stopwords NLTK
python -c "import nltk; nltk.download('stopwords')"
```

## 📂 Structure du Projet

```
influencer-or-observer/
├── data/                  # Données (train.jsonl, kaggle_test.jsonl)
├── notebooks/             # Notebooks Jupyter
├── src/                   # Code source Python
├── models/                # Modèles sauvegardés
├── submissions/           # Soumissions Kaggle
├── reports/               # Rapports et figures
└── config/                # Configuration
```

## 🎯 Premier Pas

### Option 1: Notebooks Jupyter (Recommandé pour débuter)

```bash
# Lancer Jupyter
jupyter notebook

# Ouvrir dans l'ordre:
# 1. notebooks/01_EDA.ipynb          - Explorer les données
# 2. notebooks/02_baseline.ipynb     - Tester les modèles de base
# 3. notebooks/03_feature_engineering.ipynb  - Créer des features
# 4. notebooks/04_advanced_models.ipynb      - Modèles avancés
```

### Option 2: Script Python

```python
# example_usage.py
from src.data_loader import load_training_data, load_test_data, save_submission
from src.preprocessing import preprocess_dataframe
from src.models import LogisticRegressionModel

# 1. Charger les données
X_train, y_train = load_training_data('data/train.jsonl')
X_test = load_test_data('data/kaggle_test.jsonl')

# 2. Prétraiter
X_train = preprocess_dataframe(X_train)
X_test = preprocess_dataframe(X_test)

# 3. Entraîner un modèle
model = LogisticRegressionModel()
cv_results = model.cross_validate(X_train['full_text'], y_train)
print(f"Accuracy: {cv_results['mean_score']:.4f}")

model.fit(X_train['full_text'], y_train)

# 4. Prédire et soumettre
predictions = model.predict(X_test['full_text'])
save_submission(predictions, X_test['challenge_id'], 
                'submissions/my_submission.csv')
```

## 📊 Workflow Recommandé

### 1. Exploration (EDA)
- Ouvrir `notebooks/01_EDA.ipynb`
- Comprendre la distribution des données
- Identifier des patterns

### 2. Baseline
- Ouvrir `notebooks/02_baseline.ipynb`
- Établir un score de référence
- Tester Dummy Classifier et Logistic Regression

### 3. Amélioration
- Feature engineering dans `notebooks/03_feature_engineering.ipynb`
- Tester différentes features
- Analyser l'importance des features

### 4. Modèles Avancés
- Ouvrir `notebooks/04_advanced_models.ipynb`
- Tester Random Forest, XGBoost, Neural Networks
- Hyperparameter tuning
- Ensemble methods

### 5. Soumission
```python
# Dans votre notebook ou script
from src.data_loader import save_submission

save_submission(
    predictions=y_pred,
    challenge_ids=X_test['challenge_id'],
    output_path='submissions/my_submission.csv'
)
```

## 🔧 Configuration

Modifier `config/config.yaml` pour ajuster:
- Hyperparamètres des modèles
- Paramètres de preprocessing
- Chemins des fichiers

## 📝 Checklist Avant Soumission

- [ ] Les données sont bien chargées (train.jsonl, kaggle_test.jsonl)
- [ ] Le preprocessing est appliqué de façon identique sur train et test
- [ ] Le modèle est entraîné sur toutes les données d'entraînement
- [ ] Les prédictions sont au format correct (ID, Prediction)
- [ ] Le fichier de soumission a le bon format CSV

## 🆘 Problèmes Courants

### Erreur: "Module not found"
```bash
# Vérifier que l'environnement virtuel est activé
which python  # Doit pointer vers venv/bin/python

# Réinstaller les dépendances
pip install -r requirements.txt
```

### Erreur: "NLTK stopwords not found"
```bash
python -c "import nltk; nltk.download('stopwords')"
```

### Erreur: "File not found"
```bash
# Vérifier que vous êtes dans le bon répertoire
pwd  # Doit afficher: .../influencer-or-observer

# Vérifier que les fichiers de données existent
ls data/
```

## 📚 Ressources

- **README.md**: Documentation complète
- **config/config.yaml**: Configuration des hyperparamètres
- **src/**: Code source avec docstrings
- **reports/performance.md**: Résultats des modèles

## 💡 Tips

1. **Toujours valider avec cross-validation** avant de soumettre
2. **Sauvegarder les modèles** qui performent bien
3. **Documenter les expériences** dans les notebooks
4. **Versionner les soumissions** avec un nom descriptif
5. **Analyser les erreurs** pour comprendre où le modèle échoue

## 🎓 Pour Aller Plus Loin

- Tester CamemBERT (modèle de langage français)
- Créer des features textuelles avancées
- Faire de l'ensemble de modèles
- Analyser les tweets mal classés

Bonne chance avec le challenge ! 🚀
