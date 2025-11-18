"""
Module de preprocessing des données
Ce module gère le nettoyage et la transformation du texte
"""

import re
import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def extract_full_text(tweet: pd.Series) -> str:
    """
    Extrait le texte complet d'un tweet
    Les tweets peuvent être tronqués, avec le texte complet dans 'extended_tweet.full_text'
    
    Args:
        tweet: Série Pandas représentant un tweet
        
    Returns:
        Le texte complet du tweet
    """
    # Commencer avec le champ 'text' standard
    text = tweet.get('text', '')
    
    # Vérifier si le champ 'extended_tweet.full_text' existe
    if 'extended_tweet.full_text' in tweet.index and not pd.isna(tweet['extended_tweet.full_text']):
        text = tweet['extended_tweet.full_text']
    
    return text


def remove_urls(text: str) -> str:
    """
    Supprime les URLs d'un texte
    
    Args:
        text: Texte à nettoyer
        
    Returns:
        Texte sans URLs
    """
    # Pattern pour détecter les URLs (http, https, www)
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_pattern, '', text)
    
    # Pattern pour www
    www_pattern = r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(www_pattern, '', text)
    
    return text.strip()


def remove_mentions(text: str) -> str:
    """
    Supprime les mentions (@username) d'un texte
    
    Args:
        text: Texte à nettoyer
        
    Returns:
        Texte sans mentions
    """
    mention_pattern = r'@[A-Za-z0-9_]+'
    text = re.sub(mention_pattern, '', text)
    return text.strip()


def remove_hashtags(text: str) -> str:
    """
    Supprime les hashtags (#tag) d'un texte
    
    Args:
        text: Texte à nettoyer
        
    Returns:
        Texte sans hashtags
    """
    hashtag_pattern = r'#[A-Za-z0-9_]+'
    text = re.sub(hashtag_pattern, '', text)
    return text.strip()


def clean_text(text: str, 
               remove_url: bool = True,
               remove_mention: bool = True,
               remove_hashtag: bool = False,
               lowercase: bool = True,
               remove_extra_spaces: bool = True) -> str:
    """
    Nettoie un texte en appliquant plusieurs transformations
    
    Args:
        text: Texte à nettoyer
        remove_url: Si True, supprime les URLs
        remove_mention: Si True, supprime les mentions
        remove_hashtag: Si True, supprime les hashtags
        lowercase: Si True, convertit en minuscules
        remove_extra_spaces: Si True, supprime les espaces multiples
        
    Returns:
        Texte nettoyé
    """
    if not isinstance(text, str):
        return ""
    
    # Supprimer les URLs
    if remove_url:
        text = remove_urls(text)
    
    # Supprimer les mentions
    if remove_mention:
        text = remove_mentions(text)
    
    # Supprimer les hashtags
    if remove_hashtag:
        text = remove_hashtags(text)
    
    # Convertir en minuscules
    if lowercase:
        text = text.lower()
    
    # Supprimer les espaces multiples
    if remove_extra_spaces:
        text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def preprocess_dataframe(df: pd.DataFrame, 
                        text_column: str = 'full_text',
                        clean_params: Optional[dict] = None) -> pd.DataFrame:
    """
    Prétraite un DataFrame en appliquant les transformations de texte
    
    Args:
        df: DataFrame à prétraiter
        text_column: Nom de la colonne contenant le texte
        clean_params: Paramètres de nettoyage (optionnel)
        
    Returns:
        DataFrame prétraité
    """
    logger.info("Début du preprocessing du DataFrame")
    
    df = df.copy()
    
    # Extraire le texte complet si nécessaire
    if text_column not in df.columns:
        logger.info("Extraction du texte complet des tweets")
        df[text_column] = df.apply(lambda tweet: extract_full_text(tweet), axis=1)
    
    # Paramètres de nettoyage par défaut
    if clean_params is None:
        clean_params = {
            'remove_url': True,
            'remove_mention': True,
            'remove_hashtag': False,
            'lowercase': True,
            'remove_extra_spaces': True
        }
    
    # Nettoyer le texte
    logger.info(f"Nettoyage de la colonne '{text_column}'")
    df[text_column] = df[text_column].apply(lambda x: clean_text(x, **clean_params))
    
    # Supprimer les lignes avec un texte trop court
    initial_count = len(df)
    df = df[df[text_column].str.len() >= 5]
    removed_count = initial_count - len(df)
    
    if removed_count > 0:
        logger.info(f"Suppression de {removed_count} tweets trop courts")
    
    logger.info(f"Preprocessing terminé: {len(df)} tweets restants")
    
    return df


def get_text_statistics(df: pd.DataFrame, text_column: str = 'full_text') -> dict:
    """
    Calcule des statistiques sur le texte
    
    Args:
        df: DataFrame à analyser
        text_column: Nom de la colonne contenant le texte
        
    Returns:
        Dictionnaire avec les statistiques
    """
    text_lengths = df[text_column].str.len()
    word_counts = df[text_column].str.split().str.len()
    
    stats = {
        'mean_length': text_lengths.mean(),
        'median_length': text_lengths.median(),
        'min_length': text_lengths.min(),
        'max_length': text_lengths.max(),
        'mean_words': word_counts.mean(),
        'median_words': word_counts.median(),
        'min_words': word_counts.min(),
        'max_words': word_counts.max()
    }
    
    return stats


if __name__ == "__main__":
    # Exemple d'utilisation
    logging.basicConfig(level=logging.INFO)
    
    # Exemple de nettoyage de texte
    sample_text = "@user1 Check out this link http://example.com #Python #DataScience"
    print(f"Texte original: {sample_text}")
    
    cleaned = clean_text(sample_text, remove_hashtag=False)
    print(f"Texte nettoyé (hashtags conservés): {cleaned}")
    
    cleaned_no_hashtag = clean_text(sample_text, remove_hashtag=True)
    print(f"Texte nettoyé (sans hashtags): {cleaned_no_hashtag}")
