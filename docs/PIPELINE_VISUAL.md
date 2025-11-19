# 📊 Pipeline Visuel - Influencer or Observer

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    🎯 CHALLENGE KAGGLE                                   │
│              Influencer or Observer Classification                       │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  📥 DONNÉES BRUTES                                                       │
├─────────────────────────────────────────────────────────────────────────┤
│  Train: 154,914 tweets → 38,560 utilisateurs                            │
│  Test:  103,380 tweets → 25,890 utilisateurs                            │
│  Features: full_text + 194 features numériques                          │
└──────────────────────┬──────────────────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  🔧 FEATURE ENGINEERING (Section 1.5)                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  ✓ Extraction features textuelles                                       │
│    • Longueur, mots, hashtags, mentions, URLs, emojis                   │
│    • Style: majuscules, ponctuation (!, ?)                              │
│    • Nombres mentionnés                                                 │
│                                                                          │
│  ✓ Agrégation par utilisateur                                           │
│    • Statistiques: mean, std, max, sum                                  │
│    • Concat textes par utilisateur                                      │
│                                                                          │
│  ✓ Features numériques                                                  │
│    • 194 features natives du dataset                                    │
│    • Normalisation StandardScaler                                       │
└──────────────────────┬──────────────────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  🤖 MODÉLISATION                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────┐         │
│  │  📊 MODÈLES CLASSIQUES                                      │         │
│  ├────────────────────────────────────────────────────────────┤         │
│  │                                                             │         │
│  │  1. Logistic Regression (Section 2)                        │         │
│  │     • TF-IDF trigrammes (1,2,3)                            │         │
│  │     • 100K max_features                                    │         │
│  │     • GridSearchCV sur C                                   │         │
│  │     • Texte + 194 features numériques                      │         │
│  │     ⭐ Accuracy: ~XX%                                       │         │
│  │                                                             │         │
│  │  2. SVM (Section 4.5)                                      │         │
│  │     • LinearSVC avec trigrammes                            │         │
│  │     • class_weight='balanced'                              │         │
│  │     ⭐ Accuracy: ~XX%                                       │         │
│  │                                                             │         │
│  │  3. Random Forest (Section 3)                              │         │
│  │     • 200 estimateurs, max_depth=20                        │         │
│  │     • TF-IDF + features numériques                         │         │
│  │     ⭐ Accuracy: ~XX%                                       │         │
│  │                                                             │         │
│  │  4. LightGBM (Section 4)                                   │         │
│  │     • 500 estimateurs, lr=0.05                             │         │
│  │     • Features combinées                                   │         │
│  │     ⭐ Accuracy: ~XX%                                       │         │
│  │                                                             │         │
│  └────────────────────────────────────────────────────────────┘         │
│                              │                                           │
│                              ▼                                           │
│  ┌────────────────────────────────────────────────────────────┐         │
│  │  🤖 TRANSFORMERS                                            │         │
│  ├────────────────────────────────────────────────────────────┤         │
│  │                                                             │         │
│  │  5. CamemBERT (Section 8)                                  │         │
│  │     • Fine-tuned sur données françaises                    │         │
│  │     • Optimisé Mac M4 (MPS backend)                        │         │
│  │     • Mode échantillon: 15% (~10 min)                      │         │
│  │     • Mode complet: 100% (~30 min)                         │         │
│  │     ⭐ Accuracy: ~XX%                                       │         │
│  │                                                             │         │
│  └────────────────────────────────────────────────────────────┘         │
│                                                                          │
└──────────────────────┬──────────────────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  🎭 ENSEMBLES (Section 8.5)                                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────┐          ┌──────────────────────┐             │
│  │  Vote Majoritaire    │          │  Moyenne Probas      │             │
│  ├──────────────────────┤          ├──────────────────────┤             │
│  │  • 4 modèles         │          │  • 4 modèles         │             │
│  │  • Consensus         │          │  • Pondération       │             │
│  │  ⭐⭐⭐ Recommandé   │          │  ⭐⭐⭐ Recommandé   │             │
│  └──────────────────────┘          └──────────────────────┘             │
│                                                                          │
└──────────────────────┬──────────────────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  🎯 AGRÉGATION PAR UTILISATEUR (Section 7)                               │
├─────────────────────────────────────────────────────────────────────────┤
│  CRUCIAL: Kaggle évalue au niveau utilisateur, pas tweet !              │
│                                                                          │
│  Méthode 1: Moyenne des probabilités                                    │
│     tweets_user_1: [0.7, 0.8, 0.9] → mean=0.8 → Influencer              │
│                                                                          │
│  Méthode 2: Vote majoritaire                                            │
│     tweets_user_2: [1, 1, 0, 0, 1] → 3/5 → Influencer                   │
│                                                                          │
└──────────────────────┬──────────────────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  📤 SOUMISSIONS KAGGLE                                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Priorité 1 (Soumettre en premier):                                     │
│  ⭐⭐⭐ ensemble_mean_submission.csv                                     │
│  ⭐⭐⭐ ensemble_vote_submission.csv                                     │
│                                                                          │
│  Priorité 2 (Tester ensuite):                                           │
│  ⭐⭐ logistic_regression_advanced_submission.csv                        │
│  ⭐⭐ svm_advanced_submission.csv                                        │
│  ⭐⭐ random_forest_advanced_submission.csv                              │
│  ⭐⭐ lightgbm_advanced_submission.csv                                   │
│                                                                          │
│  Priorité 3 (Si temps disponible):                                      │
│  ⭐ camembert_submission.csv                                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  ⏱️  TEMPS D'EXÉCUTION (Mac M4)                                          │
├─────────────────────────────────────────────────────────────────────────┤
│  • Mode Échantillon (USE_SAMPLE=True):  10-15 min                       │
│  • Mode Complet (USE_SAMPLE=False):     30-45 min                       │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  🔧 OPTIMISATIONS MAC M4                                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  • Neural Engine 16-core                                                │
│  • GPU jusqu'à 40-core                                                  │
│  • Memory bandwidth: 546 GB/s                                           │
│  • MPS backend pour PyTorch                                             │
│  • Batch size: 32 (train) / 64 (eval)                                   │
│  • 10 threads CPU + 4 dataloader workers                                │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  📚 RÉFÉRENCES SCIENTIFIQUES                                             │
├─────────────────────────────────────────────────────────────────────────┤
│  1. TF-IDF + LogReg: 90% accuracy sur tweets (2024)                     │
│  2. SVM Trigrammes: Études comparatives                                 │
│  3. CamemBERT: RoBERTa français (Martin et al.)                         │
│  4. Class Balancing: Standard pour déséquilibre                         │
│  5. Ensembles: Approche top Kagglers                                    │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  ✅ CHECKLIST DE SOUMISSION                                              │
├─────────────────────────────────────────────────────────────────────────┤
│  [ ] Exécuter 04_advanced_models.ipynb                                  │
│  [ ] Vérifier les fichiers dans submissions/                            │
│  [ ] Soumettre ensemble_mean_submission.csv                             │
│  [ ] Noter le score Kaggle                                              │
│  [ ] Tester autres soumissions                                          │
│  [ ] Analyser les résultats                                             │
│  [ ] Optimiser hyperparamètres si besoin                                │
└─────────────────────────────────────────────────────────────────────────┘

                         🏆 BONNE CHANCE ! 🏆
```

## 📖 Documentation Disponible

- **Guide Complet** : `notebooks/ADVANCED_MODELS_GUIDE.md` (200+ lignes)
- **Quick Start** : `QUICKSTART_NEW.md` (démarrage rapide)
- **Résumé** : `IMPROVEMENTS_SUMMARY.md` (ce qui a été fait)
- **README** : `README.md` (vue d'ensemble du projet)

## 🚀 Pour Commencer

```bash
jupyter notebook notebooks/04_advanced_models.ipynb
# Menu: Cell > Run All
# Attendre 15-30 min
# Soumettre: submissions/ensemble_mean_submission.csv
```

---

*Pipeline optimisé pour Mac M4 avec Apple Silicon*
*Basé sur les meilleures pratiques de la recherche 2024*
