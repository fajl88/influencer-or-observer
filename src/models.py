"""
Module de définition et d'entraînement des modèles
Ce module contient les classes et fonctions pour créer et entraîner les modèles
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
import logging
import pickle
from pathlib import Path

# Scikit-learn imports
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

# NLTK imports
import nltk
from nltk.corpus import stopwords

# Télécharger les stopwords si nécessaire
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

logger = logging.getLogger(__name__)


class BaselineModel:
    """
    Modèle de baseline simple (Dummy Classifier)
    """
    
    def __init__(self, strategy: str = "most_frequent"):
        """
        Args:
            strategy: Stratégie du dummy classifier ('most_frequent', 'stratified', 'uniform')
        """
        self.strategy = strategy
        self.model = DummyClassifier(strategy=strategy)
        self.name = f"Baseline_{strategy}"
    
    def fit(self, X: pd.Series, y: pd.Series) -> 'BaselineModel':
        """Entraîne le modèle"""
        logger.info(f"Entraînement du modèle {self.name}")
        self.model.fit(X.values.reshape(-1, 1), y)
        return self
    
    def predict(self, X: pd.Series) -> np.ndarray:
        """Fait des prédictions"""
        return self.model.predict(X.values.reshape(-1, 1))
    
    def score(self, X: pd.Series, y: pd.Series) -> float:
        """Calcule le score"""
        return self.model.score(X.values.reshape(-1, 1), y)


class LogisticRegressionModel:
    """
    Modèle de Régression Logistique avec TF-IDF
    """
    
    def __init__(self, 
                 max_features: int = 1000,
                 max_df: float = 0.7,
                 min_df: int = 3,
                 ngram_range: Tuple[int, int] = (1, 2),
                 use_stopwords: bool = True,
                 stopwords_language: str = 'french',
                 solver: str = 'liblinear',
                 C: float = 1.0,
                 random_state: int = 42):
        """
        Args:
            max_features: Nombre maximum de features TF-IDF
            max_df: Fréquence maximale des documents
            min_df: Fréquence minimale des documents
            ngram_range: Range des n-grams
            use_stopwords: Utiliser les stopwords
            stopwords_language: Langue des stopwords
            solver: Solver pour la régression logistique
            C: Paramètre de régularisation
            random_state: Seed pour la reproductibilité
        """
        self.max_features = max_features
        self.max_df = max_df
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.use_stopwords = use_stopwords
        self.stopwords_language = stopwords_language
        self.solver = solver
        self.C = C
        self.random_state = random_state
        self.name = "LogisticRegression_TFIDF"
        
        # Utiliser les stopwords NLTK
        stop_words = None
        if use_stopwords and stopwords_language == 'french':
            stop_words = stopwords.words('french')
        
        # Créer le pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words=stop_words,
                max_df=max_df,
                min_df=min_df,
                max_features=max_features,
                ngram_range=ngram_range
            )),
            ('clf', LogisticRegression(
                random_state=random_state,
                solver=solver,
                C=C,
                max_iter=1000
            ))
        ])
    
    def fit(self, X: pd.Series, y: pd.Series) -> 'LogisticRegressionModel':
        """Entraîne le modèle"""
        logger.info(f"Entraînement du modèle {self.name}")
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, X: pd.Series) -> np.ndarray:
        """Fait des prédictions"""
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: pd.Series) -> np.ndarray:
        """Fait des prédictions probabilistes"""
        return self.pipeline.predict_proba(X)
    
    def score(self, X: pd.Series, y: pd.Series) -> float:
        """Calcule le score"""
        return self.pipeline.score(X, y)
    
    def cross_validate(self, 
                      X: pd.Series, 
                      y: pd.Series, 
                      cv: int = 5,
                      scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        Effectue une validation croisée
        
        Args:
            X: Features
            y: Target
            cv: Nombre de folds
            scoring: Métrique à utiliser
            
        Returns:
            Dictionnaire avec les résultats
        """
        logger.info(f"Validation croisée {cv}-fold en cours...")
        
        kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(self.pipeline, X, y, cv=kfold, scoring=scoring)
        
        results = {
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores)
        }
        
        logger.info(f"Score moyen: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
        
        return results


class RandomForestModel:
    """
    Modèle Random Forest avec TF-IDF
    """
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: int = 1000,
                 max_df: float = 0.7,
                 min_df: int = 3,
                 ngram_range: Tuple[int, int] = (1, 2),
                 use_stopwords: bool = True,
                 stopwords_language: str = 'french',
                 random_state: int = 42,
                 n_jobs: int = -1):
        """
        Args:
            n_estimators: Nombre d'arbres
            max_depth: Profondeur maximale des arbres
            min_samples_split: Nombre minimum d'échantillons pour split
            min_samples_leaf: Nombre minimum d'échantillons par feuille
            max_features: Nombre maximum de features TF-IDF
            max_df: Fréquence maximale des documents
            min_df: Fréquence minimale des documents
            ngram_range: Range des n-grams
            use_stopwords: Utiliser les stopwords
            stopwords_language: Langue des stopwords
            random_state: Seed pour la reproductibilité
            n_jobs: Nombre de jobs parallèles
        """
        self.name = "RandomForest_TFIDF"
        self.random_state = random_state
        
        # Utiliser les stopwords NLTK
        stop_words = None
        if use_stopwords and stopwords_language == 'french':
            stop_words = stopwords.words('french')
        
        # Créer le pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words=stop_words,
                max_df=max_df,
                min_df=min_df,
                max_features=max_features,
                ngram_range=ngram_range
            )),
            ('clf', RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                n_jobs=n_jobs
            ))
        ])
    
    def fit(self, X: pd.Series, y: pd.Series) -> 'RandomForestModel':
        """Entraîne le modèle"""
        logger.info(f"Entraînement du modèle {self.name}")
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, X: pd.Series) -> np.ndarray:
        """Fait des prédictions"""
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: pd.Series) -> np.ndarray:
        """Fait des prédictions probabilistes"""
        return self.pipeline.predict_proba(X)
    
    def score(self, X: pd.Series, y: pd.Series) -> float:
        """Calcule le score"""
        return self.pipeline.score(X, y)


def save_model(model: Any, filepath: str) -> None:
    """
    Sauvegarde un modèle avec pickle
    
    Args:
        model: Modèle à sauvegarder
        filepath: Chemin de sauvegarde
    """
    logger.info(f"Sauvegarde du modèle dans {filepath}")
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info("Modèle sauvegardé avec succès")


def load_model(filepath: str) -> Any:
    """
    Charge un modèle avec pickle
    
    Args:
        filepath: Chemin du modèle
        
    Returns:
        Modèle chargé
    """
    logger.info(f"Chargement du modèle depuis {filepath}")
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    logger.info("Modèle chargé avec succès")
    return model


if __name__ == "__main__":
    # Exemple d'utilisation
    logging.basicConfig(level=logging.INFO)
    
    # Créer des données d'exemple
    X_example = pd.Series([
        "Ceci est un tweet exemple",
        "Un autre tweet pour tester",
        "Encore un tweet"
    ])
    y_example = pd.Series([0, 1, 0])
    
    # Test du modèle de régression logistique
    print("\n=== Test Logistic Regression ===")
    lr_model = LogisticRegressionModel(max_features=100)
    lr_model.fit(X_example, y_example)
    predictions = lr_model.predict(X_example)
    print(f"Prédictions: {predictions}")
