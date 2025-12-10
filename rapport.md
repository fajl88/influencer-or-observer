# Influencer or Observer: Predicting Social Roles from Tweets

**CSC_51054_EP - Data Challenge**  
**Équipe Kaggle : [NOM DE L'ÉQUIPE]**  
**Auteurs : Malo Tamalet, [AUTRES MEMBRES]**  
**Décembre 2025**

---

## 1. Introduction et Problème

Ce challenge propose de classifier des tweets francophones selon le rôle social de leur auteur : **Influenceur** (1) ou **Observateur** (0). Les influenceurs sont caractérisés par un réseau asymétrique (beaucoup d'abonnés, peu d'abonnements), tandis que les observateurs maintiennent des relations plus réciproques. Le dataset contient 154 914 tweets d'entraînement (38 560 utilisateurs) et 103 380 tweets de test (25 890 utilisateurs). La métrique d'évaluation est l'**accuracy**.

**Contrainte critique :** Les colonnes `user.followers_count`, `user.friends_count` et `user.verified` n'existent pas dans les données. Nous avons dû construire des proxys à partir des features disponibles (notamment `user.listed_count`).

---

## 2. Preprocessing et Représentation des Données

### 2.1 Nettoyage des Textes

Nous avons appliqué le preprocessing recommandé par cardiffnlp [1] pour les modèles Twitter :
- Remplacement des @mentions par `@user`
- Remplacement des URLs par `http`
- Normalisation des espaces

### 2.2 Features Engineered (45+ features)

Nous avons extrait trois catégories de features :

| Catégorie | Exemples | Justification |
|-----------|----------|---------------|
| **Textuelles (28)** | `hashtag_count`, `emoji_count`, `has_call_to_action`, `is_reply` | Les influenceurs utilisent plus de hashtags et font plus de self-promotion |
| **Structurelles (18)** | `retweet_count`, `source_device`, `is_bot_source` | Les bots sont souvent des influenceurs |
| **Utilisateur (15)** | `user_listed_count`, `user_description_length`, `user_has_url` | `listed_count` est le meilleur proxy pour la popularité |

**Features les plus discriminantes :**
- `log_user_listed` (corrélation = 0.606) ⭐
- `user_description_length` (importance LightGBM = 736)
- `tweets_per_favourites` (importance = 699)
- `user_has_url` (Observers = 16%, Influencers = 56%)

### 2.3 Embeddings Contextuels

Nous avons généré des embeddings avec **CamemBERT** [2] :
- Extraction multi-couches (layers -1, -2, -3, -4)
- Attention pooling (meilleur que CLS token pour tweets courts)
- Dimension finale : 3072 (768 × 4 couches)

### 2.4 TF-IDF

Pour capturer les patterns n-grammes :
- Word n-grams : (1, 2), 125k features
- Char n-grams : (3, 5), 125k features
- Total : 250k features TF-IDF

---

## 3. Modélisation

### 3.1 Architecture Principale : Twitter-XLM-RoBERTa

Notre modèle principal est basé sur `cardiffnlp/twitter-xlm-roberta-base` [1], un transformer pré-entraîné sur 198M tweets multilingues.

**Améliorations apportées :**

| Technique | Description | Gain estimé |
|-----------|-------------|-------------|
| **Multi-Sample Dropout** | 5 dropout rates (0.1-0.5) moyennés | +0.5-1% |
| **Mean Pooling** | Moyenne des hidden states vs CLS | Meilleur pour tweets courts |
| **Layerwise LR Decay** (0.85) | Les couches basses apprennent plus lentement | Stabilité |
| **Label Smoothing** (0.05) | Réduit l'overconfidence | Généralisation |
| **Multi-seed** (42, 1234) | Moyenne des prédictions de 2 seeds | Variance réduite |

**Configuration d'entraînement :**
- `MAX_LEN = 224`, `EPOCHS = 4`, `BATCH_SIZE = 16`, `LR = 2e-5`
- Gradient accumulation (×2), warmup 10%
- 5-fold StratifiedKFold avec pseudo-user IDs (hash de description + profil)

### 3.2 Modèles Complémentaires

**XGBoost sur features engineered :**
```
n_estimators=2000, max_depth=10, learning_rate=0.025
GPU accelerated (tree_method='hist', device='cuda')
```

**Logistic Regression sur TF-IDF :**
```
C=4.0, max_iter=400 (baseline robuste)
```

### 3.3 Stacking / Meta-Ensemble

Nous combinons les prédictions OOF des modèles via un méta-modèle :

```
Stack Features = [transformer_oof, tfidf_oof, xgb_oof]
Meta-model = LogisticRegression(C=1.5)
+ Calibration Isotonique (réduit l'écart CV/LB)
```

### 3.4 Pseudo-Labeling

Pour les prédictions test à haute confiance (prob ≥ 0.90 ou ≤ 0.10), nous ajoutons 30% de pseudo-labels au training avec un poids réduit (0.5).

---

## 4. Optimisation et Validation

### 4.1 Adversarial Validation

Nous avons vérifié que train et test ont des distributions similaires (AUC ≈ 0.5), confirmant l'absence de data drift majeur.

### 4.2 Threshold Optimization

Le seuil optimal n'est pas 0.5. Nous optimisons sur les prédictions OOF :
```python
thresholds = np.linspace(0.35, 0.65, 31)
best_thr = argmax(accuracy_score(labels, oof >= thr))
```
Seuil typique : **0.50-0.58**

### 4.3 Hyperparameter Tuning

- **Optuna** pour les poids de l'ensemble
- **Early stopping** (patience=100 pour XGBoost)
- **Grid search** pour C de LogisticRegression

---

## 5. Résultats

| Méthode | CV Accuracy | Configuration |
|---------|-------------|---------------|
| LightGBM (features only) | ~83.8% | Baseline |
| TF-IDF + LogReg | ~82.5% | Généralise bien |
| Transformer (single seed) | ~85.0% | twitter-xlm-roberta |
| Multi-seed TF + XGB Stack | ~86-87% | Calibré |
| + Pseudo-labeling | ~87-88% | Meilleur CV |

**Soumissions générées :**
1. `submission_meta_v2.csv` (stack calibré)
2. `submission_tf_v2_thr*.csv` (transformer seul)
3. `submission_xgb_pseudo_v2.csv` (XGB + pseudo-labels)

---

## 6. Analyse et Discussion

### Ce qui fonctionne

- **Preprocessing cardiffnlp** : Essentiel pour twitter-xlm-roberta
- **Features utilisateur** : `listed_count` compense l'absence de `followers_count`
- **TF-IDF char n-grams** : Capture les patterns de style (emojis, ponctuation)
- **Calibration isotonique** : Réduit l'écart CV/LB

### Limites

- **Manque de features réseau** : Sans followers/friends, la classification repose principalement sur le contenu
- **Pseudo-labeling** : Risque d'overfitting si seuils mal choisis
- **Temps de calcul** : ~6h pour le pipeline complet (GPU RTX 4000)

### Améliorations futures

- Graph Neural Networks pour modéliser les interactions @mentions
- Ensemble de plusieurs architectures transformer (DeBERTa, etc.)
- Augmentation de données (back-translation)

---

## 7. Instructions d'Exécution

```bash
# 1. Activer l'environnement
source kaggle-env/bin/activate

# 2. Lancer le notebook principal
jupyter notebook notebooks/transformer_ultimate_v2.ipynb

# 3. Exécuter toutes les cellules (~6h sur GPU)

# 4. Soumettre
kaggle competitions submit -c influencer-or-observer \
    -f submission/submission_meta_v2.csv -m "Final submission"
```

**Dépendances :** transformers, torch, xgboost, lightgbm, scikit-learn, pandas, numpy

---

## Références

[1] Barbieri et al. (2022). *XLM-T: Multilingual Language Models in Twitter for Sentiment Analysis and Beyond*. arXiv:2104.12250

[2] Martin et al. (2020). *CamemBERT: a Tasty French Language Model*. ACL 2020

[3] Ke et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS 2017

[4] Chen & Guestrin (2016). *XGBoost: A Scalable Tree Boosting System*. KDD 2016

---

## Annexe A : Top Features par Importance

| Rang | Feature | Importance LightGBM | Corrélation |
|------|---------|---------------------|-------------|
| 1 | `user_description_length` | 736 | 0.24 |
| 2 | `tweets_per_favourites` | 699 | - |
| 3 | `user_favourites_count` | 646 | 0.18 |
| 4 | `user_statuses_count` | 639 | 0.44 |
| 5 | `user_listed_count` | 607 | 0.61 |
| 6 | `listed_per_status` | 553 | - |

## Annexe B : Architecture du Modèle Custom

```python
class MultiSampleDropoutClassifier(nn.Module):
    def __init__(self, model_name, num_labels=2, dropout_samples=5):
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropouts = [Dropout(p) for p in [0.1, 0.2, 0.3, 0.4, 0.5]]
        self.classifier = Linear(hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask, labels=None):
        hidden = self.transformer(input_ids, attention_mask).last_hidden_state
        pooled = mean_pooling(hidden, attention_mask)
        logits = mean([self.classifier(drop(pooled)) for drop in self.dropouts])
        return CrossEntropy(logits, labels, label_smoothing=0.05)
```

## Annexe C : Distribution des Prédictions

| Modèle | % Influencers prédits | Seuil |
|--------|----------------------|-------|
| Baseline (dummy) | 53% | 0.5 |
| TF-IDF | 48% | 0.52 |
| Transformer | 46% | 0.55 |
| Meta-stack | 47% | 0.53 |
