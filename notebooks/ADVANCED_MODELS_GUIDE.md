# 🚀 Guide des Modèles Avancés

## Vue d'ensemble

Ce notebook implémente les meilleures pratiques de la recherche scientifique récente pour la classification de textes (tweets) en français, visant à distinguer les **Influenceurs** des **Observateurs**.

## 🎯 Améliorations Implémentées

### 1. Feature Engineering Avancé
**Section 1.5** - Extraction de features textuelles et numériques

- ✅ **Features textuelles extraites**:
  - Longueur du texte et nombre de mots
  - Comptage: hashtags, mentions, URLs, emojis
  - Style d'écriture: ratio de majuscules, ponctuation (!, ?)
  - Nombres (stats souvent mentionnées par influenceurs)

- ✅ **Features numériques**:
  - Utilisation des **194 features** natives du dataset
  - Agrégation par utilisateur (mean, std, max, sum)
  - Normalisation avec StandardScaler

### 2. Modèles Baseline Optimisés

#### Logistic Regression + TF-IDF
**Section 2** - Baseline améliorée basée sur la recherche

```python
# Configuration optimale
TfidfVectorizer(
    ngram_range=(1, 3),      # Trigrammes recommandés
    max_features=100000,      # Large vocabulaire
    min_df=2,                # Filtrer mots rares
    sublinear_tf=True        # Log-scaling
)

LogisticRegression(
    solver='saga',           # Meilleur pour large datasets
    class_weight='balanced'  # Gérer déséquilibre
)
```

**Basé sur**: Papier 2024 montrant 90% d'exactitude avec LogReg sur tweets

#### SVM avec Trigrammes
**Section 4.5** - Support Vector Machine

```python
LinearSVC(
    C=1.0,
    class_weight='balanced',
    max_iter=2000
)
```

**Basé sur**: Études montrant que SVM trigramme peut surpasser modèles avancés

#### Random Forest & LightGBM
**Sections 3 & 4** - Modèles d'arbres pour features numériques

- Random Forest: 200 estimateurs, max_depth=20
- LightGBM: 500 estimateurs, learning_rate=0.05
- Combinent TF-IDF (5000 features) + features numériques normalisées

### 3. Agrégation par Utilisateur
**Section 7** - Crucial pour Kaggle

```python
def aggregate_predictions_by_user(df, predictions, method='mean'):
    """
    Méthodes:
    - 'mean': Moyenne des probabilités par utilisateur
    - 'majority': Vote majoritaire des prédictions
    """
```

**Pourquoi**: L'évaluation Kaggle se fait au niveau utilisateur, pas tweet

### 4. Transformers (CamemBERT)
**Section 8** - Fine-tuning optimisé pour Mac M4

#### Mode Échantillon (Défaut)
- `USE_SAMPLE = True`
- 15% des données pour test rapide (~5-10 min)
- Validation du pipeline avant lancement complet

#### Mode Complet
- `USE_SAMPLE = False`
- Toutes les données (~20-40 min sur M4)
- Pour soumission finale

#### Optimisations M4
```python
# Configuration spécifique M4
torch.set_num_threads(10)        # 10 cœurs M4
device = "mps"                   # Apple Silicon GPU
per_device_train_batch_size=32   # Gros batch grâce à mémoire unifiée
dataloader_num_workers=4         # Parallélisme
```

**Spécifications M4 Max**:
- Neural Engine 16-core
- GPU jusqu'à 40-core
- Memory bandwidth jusqu'à 546 GB/s
- Architecture idéale pour ML

### 5. Ensemble de Modèles
**Section 8.5** - Combiner plusieurs modèles

#### Vote Majoritaire
```python
# Si 3/4 modèles prédisent "Influenceur" → Influenceur
ensemble_vote = (predictions.sum(axis=0) >= n_models / 2)
```

#### Moyenne de Probabilités
```python
# Moyenne des probabilités de tous les modèles
ensemble_mean = model_probas.mean(axis=0)
```

**Avantage**: Plus robuste que modèles individuels

## 📊 Pipeline d'Exécution Recommandé

### Exécution Rapide (Test)
1. Exécuter cellules 1 à 7 → Modèles classiques (~5-10 min)
2. Exécuter cellule 8 avec `USE_SAMPLE=True` → CamemBERT test (~5 min)
3. Exécuter section 8.5 → Ensembles
4. Soumettre `ensemble_mean_submission.csv`

### Exécution Complète (Production)
1. Exécuter cellules 1 à 7 → Modèles classiques (~15-20 min)
2. **Modifier** cellule 8: `USE_SAMPLE = False`
3. Exécuter cellule 8 → CamemBERT complet (~20-40 min)
4. Exécuter section 8.5 → Ensembles incluant CamemBERT
5. Soumettre tous les fichiers et comparer

## 🏆 Stratégie de Soumission Kaggle

### Ordre de Priorité
1. **`ensemble_mean_submission.csv`** ⭐⭐⭐
   - Combine tous les modèles classiques
   - Plus robuste aux variations

2. **`ensemble_vote_submission.csv`** ⭐⭐⭐
   - Vote majoritaire
   - Alternative à la moyenne

3. **Meilleur modèle individuel** ⭐⭐
   - Checker le graphique de comparaison
   - Souvent LightGBM ou Logistic Regression

4. **`camembert_submission.csv`** ⭐⭐
   - Si entraîné en mode complet
   - Potentiellement le meilleur

### Conseils Kaggle
- 📤 Vous pouvez soumettre **5 fois par jour**
- 📊 Testez plusieurs soumissions pour voir ce qui marche
- 🔄 Alternez entre ensembles et modèles individuels
- 📝 Notez les scores pour chaque approche

## 🔧 Dépannage

### Erreur Mémoire CamemBERT
```python
# Réduire batch size dans section 8
train_batch_size = 8  # au lieu de 32
eval_batch_size = 16  # au lieu de 64
```

### Entraînement trop lent
```python
# Augmenter l'échantillon progressivement
SAMPLE_SIZE = 0.15  # 15%
SAMPLE_SIZE = 0.30  # 30%
SAMPLE_SIZE = 0.50  # 50%
```

### Grid Search trop long
```python
# Réduire le grid
param_grid_lr = {
    'clf__C': [1.0, 10.0],  # Au lieu de [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
}
```

## 📚 Références Scientifiques

Les implémentations sont basées sur:

1. **TF-IDF + Logistic Regression**: Papier 2024 montrant 90% d'exactitude sur tweets
2. **SVM Trigramme**: Études comparatives sur classification de texte
3. **CamemBERT**: Modèle RoBERTa français battant mBERT et XLM-R
4. **Class Balancing**: Pratique standard pour datasets déséquilibrés
5. **Ensemble Methods**: Approche classique des top Kagglers

## 🎓 Apprentissages Clés

### Ce qui marche bien
✅ TF-IDF avec trigrammes (1,2,3)
✅ Class balancing avec `class_weight='balanced'`
✅ Features numériques normalisées
✅ Agrégation par utilisateur
✅ Ensembles de modèles

### Ce qui est moins important
❌ Modèles très complexes sur petit dataset
❌ Trop de features (overfitting)
❌ Ignorer le déséquilibre de classes
❌ Oublier l'agrégation par utilisateur

## 🚀 Prochaines Améliorations Possibles

### Court terme
- [ ] Tester XLM-RoBERTa (multilingue)
- [ ] RandomizedSearchCV pour LightGBM
- [ ] Analyse des erreurs (où les modèles se trompent)

### Moyen terme
- [ ] Data augmentation (back-translation)
- [ ] Stacking avec meta-modèle
- [ ] Features sociales avancées (graphe d'influence)

### Long terme
- [ ] Fine-tuning avec plusieurs runs (ensembles de transformers)
- [ ] Pseudo-labeling du test set
- [ ] Architecture custom (CNN + LSTM sur embeddings)

## 💻 Utilisation de votre Mac M4

Votre Mac M4 est **parfaitement adapté** pour ce challenge:

### Avantages
- ✅ MPS backend pour PyTorch (GPU Apple Silicon)
- ✅ Mémoire unifiée (pas de transfert CPU↔GPU)
- ✅ 10 cœurs pour parallélisation
- ✅ Neural Engine pour accélération ML

### Optimisations appliquées
```python
torch.set_num_threads(10)              # Utiliser tous les cœurs
device = torch.device("mps")           # GPU Apple Silicon
per_device_train_batch_size=32         # Gros batch possible
dataloader_num_workers=4               # Parallélisme I/O
```

### Benchmarks attendus
- Modèles classiques: **5-15 min** (tous ensemble)
- CamemBERT échantillon: **5-10 min**
- CamemBERT complet: **20-40 min** (3 epochs)

---

**Bonne chance pour le challenge ! 🏆**

Pour toute question, référez-vous aux commentaires dans le code du notebook.
