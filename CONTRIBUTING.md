# 🤝 Guide de Contribution

Merci de votre intérêt pour contribuer au projet **Influencer or Observer** !

## 🎯 Comment Contribuer

### 1. Avant de Commencer

- Lire le [README.md](README.md) et [QUICKSTART.md](QUICKSTART.md)
- Exécuter `python check_setup.py` pour vérifier votre environnement
- Explorer les notebooks dans l'ordre (01 → 04)

### 2. Types de Contributions

#### 🐛 Corrections de Bugs
- Créer une issue décrivant le bug
- Forker le repo et créer une branche `fix/description-du-bug`
- Corriger et tester
- Soumettre une Pull Request

#### ✨ Nouvelles Features
- Proposer la feature dans une issue
- Attendre validation avant de commencer
- Créer une branche `feature/nom-de-la-feature`
- Implémenter avec tests et documentation
- Soumettre une Pull Request

#### 📚 Documentation
- Améliorer les docstrings
- Ajouter des exemples
- Corriger les typos
- Traduire (si pertinent)

#### 🧪 Expérimentations
- Tester de nouveaux modèles dans `notebooks/`
- Documenter les résultats dans `reports/`
- Partager les insights dans une issue

## 📋 Standards de Code

### Python

#### Style
```python
# ✅ Bon
def extract_features(text: str, config: dict) -> pd.DataFrame:
    """
    Extrait les features d'un texte.
    
    Args:
        text: Le texte à analyser
        config: Configuration des paramètres
        
    Returns:
        DataFrame avec les features extraites
    """
    # Code ici
    pass

# ❌ Éviter
def extract(t,c):
    # Code sans documentation
    pass
```

#### Conventions
- **Noms de variables**: `snake_case`
- **Noms de classes**: `PascalCase`
- **Constantes**: `UPPER_SNAKE_CASE`
- **Imports**: Groupés et triés (stdlib → tiers → local)
- **Docstrings**: Google style pour toutes les fonctions publiques
- **Type hints**: Obligatoires pour les fonctions principales

#### Formatage
```bash
# Utiliser black pour le formatage
black src/ --line-length 100

# Utiliser isort pour les imports
isort src/
```

### Notebooks

#### Structure
```markdown
# Titre Principal

## Objectif
Description claire de ce que fait le notebook

[Code Cell 1: Imports]
[Code Cell 2: Chargement des données]
[Markdown: Explication de la section suivante]
[Code Cell 3: Analyse]
...

## Conclusions
Résumé des insights
```

#### Bonnes Pratiques
- Nettoyer les outputs avant commit (optionnel)
- Utiliser des titres Markdown clairs
- Commenter le code complexe
- Sauvegarder les figures importantes dans `reports/figures/`
- Documenter les expériences ratées (learning!)

## 🧪 Tests

### Avant de Soumettre
```bash
# 1. Vérifier que tout fonctionne
python check_setup.py

# 2. Tester les imports
python -c "from src.models import *; print('OK')"

# 3. Vérifier qu'il n'y a pas d'erreurs
# (dans VS Code, vérifier la vue "Problèmes")

# 4. Tester le notebook modifié
jupyter notebook notebooks/votre_notebook.ipynb
```

## 📝 Messages de Commit

### Format
```
<type>: <description courte>

<description détaillée si nécessaire>
```

### Types
- `feat`: Nouvelle fonctionnalité
- `fix`: Correction de bug
- `docs`: Documentation
- `style`: Formatage (sans changement de logique)
- `refactor`: Refactoring du code
- `test`: Ajout de tests
- `chore`: Tâches diverses (config, etc.)

### Exemples
```bash
feat: ajouter feature engineering pour sentiment analysis

- Implémenter analyse de sentiment avec VADER
- Ajouter tests unitaires
- Documenter dans 03_feature_engineering.ipynb
```

```bash
fix: corriger bug dans load_training_data

Le parsing des tweets tronqués ne récupérait pas le texte complet.
Utilise maintenant extended_tweet.full_text en priorité.
```

```bash
docs: améliorer README avec exemples d'utilisation
```

## 🔍 Processus de Review

### Avant la Pull Request
- [ ] Code formaté (black, isort)
- [ ] Docstrings ajoutées
- [ ] Tests passent
- [ ] Documentation mise à jour
- [ ] Pas de fichiers inutiles (`.pyc`, `__pycache__`, etc.)

### Checklist PR
- [ ] Description claire du changement
- [ ] Référence à l'issue (si applicable)
- [ ] Screenshots si UI/visualisation
- [ ] Tests ajoutés/modifiés
- [ ] Documentation mise à jour

## 🌳 Workflow Git

```bash
# 1. Cloner le repo
git clone https://github.com/[username]/influencer-or-observer.git
cd influencer-or-observer

# 2. Créer une branche
git checkout -b feature/ma-nouvelle-feature

# 3. Faire des commits réguliers
git add .
git commit -m "feat: ajouter nouvelle feature"

# 4. Pousser la branche
git push origin feature/ma-nouvelle-feature

# 5. Créer une Pull Request sur GitHub
```

## 📊 Ajout de Nouveaux Modèles

### Dans src/models.py
```python
class NouveauModele:
    """
    Description du modèle
    """
    
    def __init__(self, param1: int = 100, param2: float = 0.1):
        """
        Args:
            param1: Description
            param2: Description
        """
        self.name = "NouveauModele"
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer()),
            ('clf', VotreClassifier())
        ])
    
    def fit(self, X: pd.Series, y: pd.Series) -> 'NouveauModele':
        """Entraîne le modèle"""
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, X: pd.Series) -> np.ndarray:
        """Fait des prédictions"""
        return self.pipeline.predict(X)
```

### Dans notebooks/
- Tester dans `04_advanced_models.ipynb`
- Documenter les résultats
- Comparer avec les autres modèles
- Mettre à jour `reports/performance.md`

## 🎓 Ressources

### Documentation Officielle
- [Pandas](https://pandas.pydata.org/docs/)
- [Scikit-learn](https://scikit-learn.org/)
- [NLTK](https://www.nltk.org/)
- [Transformers](https://huggingface.co/docs/transformers/)

### Tutoriels
- [Kaggle Learn](https://www.kaggle.com/learn)
- [Fast.ai](https://www.fast.ai/)
- [Deep Learning Book](https://www.deeplearningbook.org/)

## 💬 Communication

### Questions
- Ouvrir une issue avec le label `question`
- Décrire clairement le problème
- Fournir un exemple reproductible si possible

### Discussions
- Utiliser GitHub Discussions pour les idées générales
- Issues pour les bugs/features spécifiques

## 🙏 Reconnaissance

Tous les contributeurs seront ajoutés au README.md !

---

**Merci de contribuer au projet ! 🚀**
