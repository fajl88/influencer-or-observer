"""
Module de fonctions utilitaires
Ce module contient des fonctions helper générales
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


def setup_logging(log_file: Optional[str] = None, level: str = 'INFO') -> None:
    """
    Configure le logging
    
    Args:
        log_file: Chemin du fichier de log (optionnel)
        level: Niveau de logging ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, level),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=getattr(logging, level),
            format=log_format
        )


def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """
    Charge le fichier de configuration JSON (converti depuis YAML)
    
    Args:
        config_path: Chemin vers le fichier de configuration
        
    Returns:
        Dictionnaire avec la configuration
    """
    # Convertir .yaml en .json pour le chemin
    json_path = config_path.replace('.yaml', '.json')
    if not Path(json_path).exists():
        json_path = config_path  # Essayer le chemin original
    
    with open(json_path, 'r') as f:
        config = json.load(f)
    return config


def save_json(data: dict, filepath: str) -> None:
    """
    Sauvegarde un dictionnaire en JSON
    
    Args:
        data: Dictionnaire à sauvegarder
        filepath: Chemin du fichier de sortie
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> dict:
    """
    Charge un fichier JSON
    
    Args:
        filepath: Chemin du fichier JSON
        
    Returns:
        Dictionnaire avec les données
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def get_class_distribution(y: pd.Series) -> Dict[str, Any]:
    """
    Calcule la distribution des classes
    
    Args:
        y: Série avec les étiquettes
        
    Returns:
        Dictionnaire avec les statistiques de distribution
    """
    counts = y.value_counts()
    total = len(y)
    
    distribution = {
        'total': total,
        'class_counts': counts.to_dict(),
        'class_percentages': (counts / total * 100).to_dict(),
        'is_balanced': (counts.max() / counts.min()) < 1.5
    }
    
    return distribution


def create_directory_structure(base_path: str = '.') -> None:
    """
    Crée la structure de dossiers du projet
    
    Args:
        base_path: Chemin de base du projet
    """
    directories = [
        'data',
        'notebooks',
        'src',
        'models',
        'submissions',
        'reports',
        'reports/figures',
        'config'
    ]
    
    for directory in directories:
        Path(base_path) / directory.mkdir(parents=True, exist_ok=True)
    
    print(f"Structure de dossiers créée dans {base_path}")


def format_elapsed_time(seconds: float) -> str:
    """
    Formate un temps écoulé en secondes en format lisible
    
    Args:
        seconds: Temps en secondes
        
    Returns:
        String formaté (ex: "2h 30m 15s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def memory_usage(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcule l'utilisation mémoire d'un DataFrame
    
    Args:
        df: DataFrame à analyser
        
    Returns:
        Dictionnaire avec les statistiques d'utilisation mémoire
    """
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    
    usage = {
        'total_mb': memory_mb,
        'per_column_mb': (df.memory_usage(deep=True) / 1024**2).to_dict(),
        'shape': df.shape
    }
    
    return usage


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Réduit l'utilisation mémoire d'un DataFrame en optimisant les types de données
    
    Args:
        df: DataFrame à optimiser
        verbose: Afficher les informations
        
    Returns:
        DataFrame optimisé
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    if verbose:
        print(f'Mémoire avant: {start_mem:.2f} MB')
        print(f'Mémoire après: {end_mem:.2f} MB')
        print(f'Réduction: {100 * (start_mem - end_mem) / start_mem:.1f}%')
    
    return df


def print_separator(char: str = '=', length: int = 50) -> None:
    """
    Affiche un séparateur
    
    Args:
        char: Caractère à utiliser
        length: Longueur du séparateur
    """
    print(char * length)


def print_section(title: str, char: str = '=', length: int = 50) -> None:
    """
    Affiche un titre de section
    
    Args:
        title: Titre de la section
        char: Caractère pour le séparateur
        length: Longueur du séparateur
    """
    print_separator(char, length)
    print(title.center(length))
    print_separator(char, length)


if __name__ == "__main__":
    # Exemple d'utilisation
    
    # Configuration du logging
    setup_logging(level='INFO')
    logger = logging.getLogger(__name__)
    
    # Test de formatage du temps
    print(f"\nTemps écoulé: {format_elapsed_time(7325)}")
    
    # Test de distribution de classes
    y = pd.Series([0, 1, 1, 0, 1, 0, 0, 1, 1, 1])
    dist = get_class_distribution(y)
    print(f"\nDistribution: {dist}")
    
    # Test de section
    print_section("TEST SECTION")
