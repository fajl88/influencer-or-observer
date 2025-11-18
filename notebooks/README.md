# Notebooks Jupyter

Ce dossier contient les notebooks Jupyter pour l'exploration, l'expérimentation et la modélisation.

## Organisation

### 01_EDA.ipynb
Analyse exploratoire des données (Exploratory Data Analysis)
- Distribution des classes
- Statistiques descriptives
- Visualisations
- Analyse du texte
- Insights pour la modélisation

### 02_baseline.ipynb  
Modèles de baseline
- Dummy Classifier
- Logistic Regression avec TF-IDF
- Évaluation initiale

### 03_feature_engineering.ipynb
Création et sélection de features
- Features textuelles avancées
- Features utilisateur
- Analyse d'importance
- Tests de différentes combinaisons

### 04_advanced_models.ipynb
Modèles avancés et optimisation
- Random Forest, XGBoost, LightGBM
- Neural Networks
- Fine-tuning de transformers (CamemBERT)
- Hyperparameter tuning
- Ensemble methods

## Convention de nommage

Format: `{numéro}_{description}.ipynb`

## Best Practices

- Utiliser des titres Markdown clairs
- Documenter les expériences et résultats
- Sauvegarder les figures importantes dans `reports/figures/`
- Sauvegarder les modèles dans `models/`
- Nettoyer les outputs avant commit (optionnel)
