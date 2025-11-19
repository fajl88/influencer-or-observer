"""
Module de chargement des données
Ce module gère le chargement des fichiers JSONL et la transformation en DataFrames
"""

import json
import pandas as pd
from pandas import json_normalize
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_jsonl(file_path: str) -> pd.DataFrame:
    """
    Charge un fichier JSONL et le convertit en DataFrame
    
    Args:
        file_path: Chemin vers le fichier JSONL
        
    Returns:
        DataFrame avec les données normalisées
    """
    logger.info(f"Chargement du fichier: {file_path}")
    
    try:
        # Charger le fichier JSONL
        data = pd.read_json(file_path, lines=True)
        
        # Normaliser les données JSON imbriquées
        data = json_normalize(data.to_dict(orient='records'))
        
        logger.info(f"Fichier chargé avec succès: {data.shape[0]} lignes, {data.shape[1]} colonnes")
        
        return data
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement du fichier {file_path}: {str(e)}")
        raise


def load_training_data(file_path: str = 'data/train.jsonl') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Charge les données d'entraînement et sépare les features de la cible
    
    Args:
        file_path: Chemin vers le fichier d'entraînement
        
    Returns:
        Tuple (X_train, y_train) avec les features et la cible
    """
    logger.info("Chargement des données d'entraînement")
    
    # Charger les données
    train_data = load_jsonl(file_path)
    
    # Vérifier que la colonne 'label' existe
    if 'label' not in train_data.columns:
        raise ValueError("La colonne 'label' n'existe pas dans les données d'entraînement")
    
    # Séparer les features de la cible
    X_train = train_data.drop('label', axis=1)
    y_train = train_data['label']
    
    # Informations sur la distribution des classes
    class_counts = y_train.value_counts()
    logger.info(f"Distribution des classes:")
    logger.info(f"  - Classe 0 (Observers): {class_counts.get(0, 0)} ({class_counts.get(0, 0)/len(y_train)*100:.2f}%)")
    logger.info(f"  - Classe 1 (Influencers): {class_counts.get(1, 0)} ({class_counts.get(1, 0)/len(y_train)*100:.2f}%)")
    
    return X_train, y_train


def load_test_data(file_path: str = 'data/kaggle_test.jsonl') -> pd.DataFrame:
    """
    Charge les données de test Kaggle
    
    Args:
        file_path: Chemin vers le fichier de test
        
    Returns:
        DataFrame avec les données de test
    """
    logger.info("Chargement des données de test")
    
    # Charger les données
    X_test = load_jsonl(file_path)
    
    return X_test


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Retourne des informations sur le dataset
    
    Args:
        df: DataFrame à analyser
        
    Returns:
        Dictionnaire avec les informations sur le dataset
    """
    info = {
        'n_samples': len(df),
        'n_features': len(df.columns),
        'features': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.to_dict()
    }
    
    return info


def save_submission(challenge_ids: pd.Series,
                   predictions: pd.Series, 
                   output_path: str = 'submissions/submission.csv') -> None:
    """
    Crée un fichier de soumission Kaggle au format requis
    
    Args:
        challenge_ids: IDs des challenges (colonne ID)
        predictions: Prédictions (0 ou 1) (colonne Prediction)
        output_path: Chemin de sortie du fichier CSV
    """
    logger.info(f"Création du fichier de soumission: {output_path}")
    
    # Créer le DataFrame de soumission
    submission = pd.DataFrame({
        'ID': challenge_ids,
        'Prediction': predictions
    })
    
    # Sauvegarder en CSV
    submission.to_csv(output_path, index=False)
    
    logger.info(f"Fichier de soumission créé avec {len(submission)} prédictions")
    logger.info(f"Distribution des prédictions:")
    logger.info(f"  - Classe 0: {(predictions == 0).sum()}")
    logger.info(f"  - Classe 1: {(predictions == 1).sum()}")


if __name__ == "__main__":
    # Exemple d'utilisation
    logging.basicConfig(level=logging.INFO)
    
    # Charger les données d'entraînement
    X_train, y_train = load_training_data()
    print(f"\nDonnées d'entraînement: {X_train.shape}")
    print(f"Premières colonnes: {X_train.columns[:10].tolist()}")
    
    # Charger les données de test
    X_test = load_test_data()
    print(f"\nDonnées de test: {X_test.shape}")
    
    # Informations sur le dataset
    info = get_dataset_info(X_train)
    print(f"\nNombre de features: {info['n_features']}")
    print(f"Nombre de valeurs manquantes: {sum(info['missing_values'].values())}")
