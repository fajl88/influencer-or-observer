# Résumé des Performances des Modèles

Ce document résume les performances des différents modèles testés sur le challenge "Influencer or Observer".

## Métrique d'Évaluation

**Métrique principale**: Accuracy (proportion de prédictions correctes)

## Résultats

### Modèles de Baseline

| Modèle | Accuracy (CV) | Accuracy (Test) | Notes |
|--------|---------------|-----------------|-------|
| Dummy (Most Frequent) | ~50% | À compléter | Prédit toujours la classe majoritaire |
| Dummy (Stratified) | ~50% | À compléter | Prédit selon la distribution des classes |

### Modèles Classiques

| Modèle | Accuracy (CV) | Accuracy (Test) | Hyperparamètres | Notes |
|--------|---------------|-----------------|-----------------|-------|
| Logistic Regression + TF-IDF | À compléter | À compléter | max_features=1000, ngrams=(1,2) | Baseline principal |
| Random Forest + TF-IDF | À compléter | À compléter | n_estimators=100 | |
| XGBoost + TF-IDF | À compléter | À compléter | | |
| LightGBM + TF-IDF | À compléter | À compléter | | |

### Modèles Avancés

| Modèle | Accuracy (CV) | Accuracy (Test) | Notes |
|--------|---------------|-----------------|-------|
| Neural Network | À compléter | À compléter | |
| CamemBERT (fine-tuned) | À compléter | À compléter | Modèle de langage pré-entraîné en français |
| Ensemble | À compléter | À compléter | Combinaison de plusieurs modèles |

## Analyse des Features

### Features les plus importantes

À compléter après l'analyse des modèles.

### Features textuelles performantes

- TF-IDF des unigrams et bigrams
- Longueur du texte
- Nombre de hashtags
- ...

### Features utilisateur performantes

- `user.statuses_count`
- Type de source
- ...

## Observations

### Points forts
- À compléter

### Points faibles
- À compléter

### Pistes d'amélioration
- [ ] Feature engineering plus poussé
- [ ] Hyperparameter tuning
- [ ] Utilisation de modèles de langage pré-entraînés
- [ ] Ensemble de modèles
- [ ] Analyse des erreurs et data augmentation

## Soumissions Kaggle

| Date | Modèle | Score Public | Score Private | Fichier |
|------|--------|--------------|---------------|---------|
| À compléter | | | | |

## Notes

- Cross-validation: 5-Fold Stratified
- Random state: 42 (pour reproductibilité)
- Language: Français (stopwords, preprocessing)

---

**Dernière mise à jour**: [Date]
