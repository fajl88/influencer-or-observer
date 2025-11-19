# 🚀 Quick Start Guide - Version Optimisée

Commencez rapidement avec les modèles avancés et générez vos soumissions Kaggle !

## ⚡ Démarrage Ultra-Rapide (5 min)

### 1. Installation

```bash
# Cloner le repo (si pas déjà fait)
git clone <repo_url>
cd influencer-or-observer

# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installer dépendances
pip install -r requirements.txt
```

### 2. Ouvrir le Notebook Principal

```bash
jupyter notebook notebooks/04_advanced_models.ipynb
```

### 3. Exécuter

**Dans Jupyter** : Menu `Cell` > `Run All`

⏱️ **Temps estimé** : 15-30 min (Mac M4)

### 4. Récupérer les Soumissions

Les fichiers sont générés automatiquement dans `submissions/` :

- ⭐ `ensemble_mean_submission.csv` (Recommandé #1)
- ⭐ `ensemble_vote_submission.csv` (Recommandé #2)
- Et 4+ autres modèles à tester

### 5. Soumettre sur Kaggle

1. Aller sur la page du challenge Kaggle
2. Upload `ensemble_mean_submission.csv`
3. Vérifier le score
4. Tester d'autres soumissions

## 📊 Deux Modes d'Exécution

### Mode Test (Rapide) - Par Défaut

```python
# Dans la cellule CamemBERT
USE_SAMPLE = True    # ✅ Activé par défaut
SAMPLE_SIZE = 0.15   # 15% des données
```

- ⏱️ **Temps** : ~10 min total
- ✅ **But** : Valider le pipeline
- 📊 **Soumissions** : Modèles classiques seulement

### Mode Production (Complet)

```python
# Dans la cellule CamemBERT
USE_SAMPLE = False   # 🔥 Pour soumission finale
```

- ⏱️ **Temps** : ~30-40 min total
- ✅ **But** : Meilleurs résultats
- 📊 **Soumissions** : Tous les modèles incluant CamemBERT

## 🎯 Stratégie Recommandée

### Jour 1 : Test Rapide
1. Exécuter en mode échantillon (`USE_SAMPLE = True`)
2. Soumettre `ensemble_mean_submission.csv`
3. Noter le score Kaggle

### Jour 2 : Production
1. Modifier : `USE_SAMPLE = False`
2. Relancer le notebook (30-40 min)
3. Soumettre toutes les versions :
   - `ensemble_mean_submission.csv`
   - `ensemble_vote_submission.csv`
   - `camembert_submission.csv`
   - Meilleur modèle classique individuel

### Jour 3 : Optimisation
1. Analyser quel modèle performe le mieux
2. Ajuster hyperparamètres si nécessaire
3. Tester variations

## 📖 Documentation Complète

Pour tout savoir sur les modèles et optimisations :

**➡️ [`notebooks/ADVANCED_MODELS_GUIDE.md`](notebooks/ADVANCED_MODELS_GUIDE.md)**

## 🐍 Alternative : Script Python

Si vous préférez la ligne de commande :

```bash
# Générer toutes les soumissions
python generate_submissions.py --models all

# Uniquement les ensembles
python generate_submissions.py --models ensemble
```

> ⚠️ **Note** : Les modèles doivent d'abord être entraînés via le notebook

## 🆘 Problèmes Fréquents

### "Erreur mémoire" avec CamemBERT

```python
# Réduire le batch size dans la cellule CamemBERT
train_batch_size = 8   # Au lieu de 32
eval_batch_size = 16   # Au lieu de 64
```

### "Package manquant"

```bash
pip install <package_name>
```

### "Notebook trop lent"

1. Utiliser `USE_SAMPLE = True`
2. Réduire `SAMPLE_SIZE` à 0.10 (10%)
3. Fermer autres applications

## ✅ Checklist de Soumission

- [ ] Notebook exécuté sans erreurs
- [ ] Fichiers CSV générés dans `submissions/`
- [ ] Vérification du format (ID, Prediction)
- [ ] Upload sur Kaggle
- [ ] Score reçu

## 🏆 Tips pour Gagner

1. **Soumettez les ensembles en premier** - Plus robustes
2. **Testez plusieurs versions** - 5 soumissions/jour autorisées
3. **Notez les scores** - Pour comparer les approches
4. **Si temps limité** - Mode échantillon suffit pour bon score

## 💡 Prochaines Étapes

Après avoir soumis vos premiers résultats :

1. Analyser les prédictions incorrectes
2. Tester d'autres hyperparamètres
3. Essayer data augmentation
4. Lire le guide complet pour optimisations avancées

---

**Bonne chance ! 🚀**

Questions ? Consultez [`notebooks/ADVANCED_MODELS_GUIDE.md`](notebooks/ADVANCED_MODELS_GUIDE.md)
