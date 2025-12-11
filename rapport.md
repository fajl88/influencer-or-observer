# Influencer or Observer : Rapport de Data Challenge

**CSC_51054_EP - Machine Learning**  
**Auteurs : Malo Tamalet, François Löning, Khalid Lamrini**  
**Décembre 2025**

---

## 1. Introduction

**Objectif :** Classifier des tweets francophones selon le rôle social de leur auteur — **Influenceur** (1) ou **Observateur** (0).

**Données :**
- Train : 154 914 tweets (38 560 utilisateurs)
- Test : 103 380 tweets (25 890 utilisateurs)
- Métrique : Accuracy

**Meilleur score obtenu : 0.847**

---

## 2. Notre Parcours : Erreurs, Apprentissages et Améliorations

### 📊 Phase 1 : La Baseline — Comprendre le Problème (Score : 0.625)

Nous avons commencé avec la baseline fournie : TF-IDF + Logistic Regression.

```
logistic_regression.csv → 0.625
```

**Notre raisonnement initial :** "Le texte du tweet devrait suffire à distinguer un influenceur d'un observateur."

**Ce que nous avons appris :** 0.625 est à peine mieux que le hasard (0.53 pour la classe majoritaire). Le contenu textuel seul ne capture pas assez d'information. Il faut regarder au-delà du texte.

---

### 🔧 Phase 2 : Feature Engineering — Explorer les Données (Score : 0.82-0.83)

**Notre hypothèse :** Les métadonnées utilisateur (activité, popularité) sont probablement plus discriminantes que le texte seul.

Nous avons analysé les colonnes disponibles et découvert des features prometteuses :
- `user.listed_count` : nombre de listes où l'utilisateur apparaît → **proxy pour la popularité**
- `user.statuses_count` : nombre de tweets publiés → **mesure d'activité**
- `source` : appareil/application utilisé → **les bots sont souvent des influenceurs**

**Première erreur :** Nous avons cherché `followers_count` et `friends_count` pendant un moment... avant de réaliser que **ces colonnes n'existent pas dans le dataset**. Leçon : toujours commencer par `df.columns` !

**Résultats :**

| Modèle | Score | Notre analyse |
|--------|-------|---------------|
| LightGBM | 0.822 | Bon, mais pas le meilleur |
| XGBoost | 0.835 | ⭐ Meilleur modèle individuel |
| Neural Network | 0.804 | Décevant, pourquoi ? |

**Pourquoi le NN sous-performe ?** Nous avons compris plus tard : overfitting massif. Le NN mémorisait les données d'entraînement au lieu de généraliser.

---

### 🎯 Phase 3 : Ensembles — Nos Échecs Instructifs (Score : 0.70-0.83)

**Notre raisonnement :** "Combinons nos modèles pour avoir le meilleur des deux mondes !"

| Méthode | Score | Ce qui s'est passé |
|---------|-------|-------------------|
| Voting majoritaire | **0.703** | ❌ CATASTROPHE |
| Moyenne simple | 0.805 | Médiocre |
| Stacking | 0.831 | Correct |

**L'échec du voting (0.703) nous a surpris.** Comment un ensemble peut-il être PIRE que chaque modèle individuel ?

**Notre analyse :** Le voting majoritaire échoue quand les modèles font des erreurs corrélées. Si 2 modèles sur 3 se trompent sur les mêmes exemples difficiles, le vote amplifie l'erreur au lieu de la corriger.

**Leçon apprise :** Ne jamais utiliser le voting aveuglément. Préférer les moyennes pondérées des probabilités, qui permettent de nuancer les prédictions.

---

### ⚙️ Phase 4 : Diagnostiquer et Corriger l'Overfitting (Score : 0.84)

**Le problème :** Notre Neural Network avait ~95% accuracy en train mais seulement ~80% en validation. Écart de 15 points = overfitting sévère.

**Notre démarche de debugging :**

1. **Hypothèse 1 :** "Le modèle est trop complexe" → Réduit les couches. Résultat : légère amélioration.

2. **Hypothèse 2 :** "Le dropout est trop faible" → Augmenté de 0.1 à 0.4. Résultat : amélioration notable.

3. **Hypothèse 3 :** "Le weight decay est insuffisant" → Augmenté de 0.01 à **4.0** (×400 !). Résultat : **0.843** 🎯

```
submission_4_weight_decay.csv → 0.843
```

**Ce que nous avons compris :** Un weight decay agressif force le modèle à garder des poids petits, ce qui limite sa capacité à mémoriser le bruit des données d'entraînement.

**Ensemble pondéré après régularisation :**
```
Weighted average (XGBoost 50% + LightGBM 30% + NN 20%) → 0.842
```

**Observation intéressante :** L'ensemble (0.842) est légèrement moins bon que le NN seul bien régularisé (0.843). Parfois, un bon modèle unique bat un ensemble médiocre.

---

### 🚀 Phase 5 : Le Saut Qualitatif — Fine-tuning Transformer (Score : 0.847)

**Notre constat :** Nous stagnions à ~0.84. Les approches classiques avaient atteint leur limite.

**Notre hypothèse :** Un modèle pré-entraîné spécifiquement sur des tweets devrait mieux comprendre le langage Twitter (abréviations, emojis, hashtags, ton).

**Choix du modèle :** `cardiffnlp/twitter-xlm-roberta-base` — pré-entraîné sur 198 millions de tweets multilingues.

**Erreur évitée :** Nous avons failli utiliser CamemBERT (français généraliste) ou BERT de base. Le modèle spécialisé Twitter s'est avéré crucial.

**Techniques appliquées (inspirées des labs du cours) :**
- **Multi-Sample Dropout :** Moyenne de 5 dropout différents (0.1 à 0.5) pour régulariser
- **Mean Pooling :** Moyenne des tokens au lieu du token [CLS], meilleur pour les textes courts
- **Preprocessing Twitter :** Normalisation des @mentions et URLs comme recommandé par les auteurs du modèle

**Résultats transformer seul :**

| Seuil de décision | Score |
|-------------------|-------|
| 0.50 (défaut) | 0.834 |
| 0.55 | 0.831 |
| 0.60 | 0.833 |

**Observation :** Le seuil par défaut (0.5) est quasi-optimal. Pas de gain significatif à optimiser le seuil ici.

**Le blend final — notre meilleure idée :**

Plutôt que de choisir entre transformer ET gradient boosting, pourquoi ne pas combiner les deux ?

```
Transformer (texte) + XGBoost (features) + TF-IDF (n-grammes)
     ↓                    ↓                      ↓
   Stacking avec Logistic Regression
     ↓
submission_transformer_blend.csv → 0.847 ⭐
```

**Pourquoi ça marche :** Chaque modèle capture des aspects différents :
- Le transformer comprend la **sémantique** du texte
- XGBoost exploite les **features numériques** (activité, popularité)
- TF-IDF capture les **mots-clés** et n-grammes discriminants

---

## 3. Récapitulatif de Notre Progression

| Phase | Approche | Score | Gain | Ce qu'on a appris |
|-------|----------|-------|------|-------------------|
| 1 | Baseline TF-IDF | 0.625 | - | Le texte seul ne suffit pas |
| 2 | Feature Engineering + XGBoost | 0.835 | +0.21 | Les métadonnées sont cruciales |
| 3 | Voting majoritaire | 0.703 | -0.13 | ❌ ERREUR : le voting peut empirer les choses |
| 3 | Stacking | 0.831 | - | Les moyennes pondérées sont plus sûres |
| 4 | NN + Weight Decay ×400 | 0.843 | +0.01 | Régularisation agressive = généralisation |
| 5 | **Transformer Blend** | **0.847** | +0.004 | Combiner text + features = optimal |

**Progression totale : +0.222 points** (de 0.625 à 0.847, soit +35% relatif)

---

## 4. Nos Erreurs et Comment Nous les Avons Corrigées

| Erreur | Impact | Solution |
|--------|--------|----------|
| Chercher `followers_count` inexistant | Temps perdu | Toujours vérifier `df.columns` d'abord |
| Voting majoritaire aveugle | Score 0.703 (pire que baseline !) | Utiliser moyennes pondérées de probas |
| Dropout trop faible (0.1) | Overfitting sévère | Augmenter à 0.3-0.5 |
| Weight decay standard (0.01) | NN mémorise le bruit | Multiplier par 400 → 4.0 |
| Utiliser un BERT généraliste | Performance sous-optimale | Choisir un modèle spécialisé Twitter |

---

## 5. Conclusion : Ce Que Ce Projet Nous a Appris

1. **Toujours explorer les données avant de coder.** Nous aurions gagné du temps en regardant les colonnes disponibles dès le début.

2. **Les ensembles ne sont pas magiques.** Le voting majoritaire peut être catastrophique (0.703). Il faut comprendre pourquoi on combine des modèles.

3. **L'overfitting se combat par la régularisation, pas par la complexité.** Un weight decay ×400 a mieux fonctionné que réduire le nombre de couches.

4. **Les modèles pré-entraînés spécialisés font la différence.** `twitter-xlm-roberta-base` > CamemBERT > BERT de base pour des tweets.

5. **Le meilleur résultat vient de la complémentarité.** Transformer (sémantique) + XGBoost (features) + TF-IDF (mots-clés) = 0.847.

---

## 6. Structure du Projet

```
influencer-or-observer/
├── data/
│   ├── train.jsonl, kaggle_test.jsonl
│   └── features/
├── notebooks/
│   └── transformer_ultimate_v2.ipynb
├── feature_engineering.ipynb
├── ensemble_stacking.ipynb
├── submission/
└── rapport.md
```

---

## Références

1. Barbieri et al. (2022). *XLM-T: Multilingual Language Models in Twitter*
2. Martin et al. (2020). *CamemBERT: a Tasty French Language Model*
3. Chen & Guestrin (2016). *XGBoost: A Scalable Tree Boosting System*

---

*Décembre 2025*
