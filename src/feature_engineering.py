"""
Module de feature engineering
Ce module crée des features supplémentaires à partir des données brutes
"""

import re
import pandas as pd
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


def count_hashtags(text: str) -> int:
    """Compte le nombre de hashtags dans un texte"""
    return len(re.findall(r'#[A-Za-z0-9_]+', text))


def count_mentions(text: str) -> int:
    """Compte le nombre de mentions dans un texte"""
    return len(re.findall(r'@[A-Za-z0-9_]+', text))


def count_urls(text: str) -> int:
    """Compte le nombre d'URLs dans un texte"""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return len(re.findall(url_pattern, text))


def count_emojis(text: str) -> int:
    """
    Compte le nombre d'emojis dans un texte
    Pattern simple pour les emojis courants
    """
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return len(emoji_pattern.findall(text))


def count_capital_letters(text: str) -> int:
    """Compte le nombre de lettres majuscules"""
    return sum(1 for c in text if c.isupper())


def has_exclamation(text: str) -> int:
    """Retourne 1 si le texte contient un point d'exclamation"""
    return 1 if '!' in text else 0


def has_question(text: str) -> int:
    """Retourne 1 si le texte contient un point d'interrogation"""
    return 1 if '?' in text else 0


def get_text_length(text: str) -> int:
    """Retourne la longueur du texte"""
    return len(text)


def get_word_count(text: str) -> int:
    """Retourne le nombre de mots"""
    return len(text.split())


def get_avg_word_length(text: str) -> float:
    """Retourne la longueur moyenne des mots"""
    words = text.split()
    if not words:
        return 0.0
    return sum(len(word) for word in words) / len(words)


def is_retweet(tweet: pd.Series) -> int:
    """Vérifie si le tweet est un retweet"""
    # Vérifier si le texte commence par "RT @"
    text = tweet.get('text', '')
    if isinstance(text, str) and text.startswith('RT @'):
        return 1
    
    # Vérifier si 'retweeted_status' existe
    if 'retweeted_status' in tweet.index and not pd.isna(tweet.get('retweeted_status')):
        return 1
    
    return 0


def is_quote(tweet: pd.Series) -> int:
    """Vérifie si le tweet est une citation"""
    if 'quoted_status' in tweet.index and not pd.isna(tweet.get('quoted_status')):
        return 1
    return 0


def is_reply(tweet: pd.Series) -> int:
    """Vérifie si le tweet est une réponse"""
    if 'in_reply_to_status_id' in tweet.index and not pd.isna(tweet.get('in_reply_to_status_id')):
        return 1
    return 0


def has_media(tweet: pd.Series) -> int:
    """Vérifie si le tweet contient des médias (photos, vidéos)"""
    # Vérifier si 'entities' contient des médias
    if 'entities' in tweet.index and not pd.isna(tweet.get('entities')):
        entities = tweet.get('entities')
        if isinstance(entities, dict) and 'media' in entities:
            return 1
    
    # Vérifier extended_entities
    if 'extended_entities' in tweet.index and not pd.isna(tweet.get('extended_entities')):
        ext_entities = tweet.get('extended_entities')
        if isinstance(ext_entities, dict) and 'media' in ext_entities:
            return 1
    
    return 0


def extract_source_type(source: str) -> str:
    """
    Extrait le type de source (iPhone, Android, Web, etc.)
    """
    if pd.isna(source) or not isinstance(source, str):
        return 'unknown'
    
    source_lower = source.lower()
    
    if 'iphone' in source_lower:
        return 'iphone'
    elif 'android' in source_lower:
        return 'android'
    elif 'web' in source_lower or 'twitter.com' in source_lower:
        return 'web'
    elif 'ipad' in source_lower:
        return 'ipad'
    elif 'tweetdeck' in source_lower:
        return 'tweetdeck'
    elif 'hootsuite' in source_lower:
        return 'hootsuite'
    elif 'buffer' in source_lower:
        return 'buffer'
    else:
        return 'other'


def create_text_features(df: pd.DataFrame, text_column: str = 'full_text') -> pd.DataFrame:
    """
    Crée des features textuelles à partir d'une colonne de texte
    
    Args:
        df: DataFrame source
        text_column: Nom de la colonne contenant le texte
        
    Returns:
        DataFrame avec les nouvelles features
    """
    logger.info("Création des features textuelles")
    
    df = df.copy()
    
    # Features de comptage
    df['text_length'] = df[text_column].apply(get_text_length)
    df['word_count'] = df[text_column].apply(get_word_count)
    df['avg_word_length'] = df[text_column].apply(get_avg_word_length)
    df['hashtag_count'] = df[text_column].apply(count_hashtags)
    df['mention_count'] = df[text_column].apply(count_mentions)
    df['url_count'] = df[text_column].apply(count_urls)
    df['emoji_count'] = df[text_column].apply(count_emojis)
    df['capital_count'] = df[text_column].apply(count_capital_letters)
    
    # Features binaires
    df['has_exclamation'] = df[text_column].apply(has_exclamation)
    df['has_question'] = df[text_column].apply(has_question)
    
    # Ratios
    df['capital_ratio'] = df['capital_count'] / (df['text_length'] + 1)
    df['hashtag_ratio'] = df['hashtag_count'] / (df['word_count'] + 1)
    df['mention_ratio'] = df['mention_count'] / (df['word_count'] + 1)
    
    logger.info(f"Features textuelles créées: {len([col for col in df.columns if col.startswith(('text_', 'word_', 'hashtag_', 'mention_', 'url_', 'emoji_', 'capital_', 'has_', 'avg_'))])} features")
    
    return df


def create_tweet_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des features liées au type de tweet
    
    Args:
        df: DataFrame source
        
    Returns:
        DataFrame avec les nouvelles features
    """
    logger.info("Création des features de tweet")
    
    df = df.copy()
    
    df['is_retweet'] = df.apply(is_retweet, axis=1)
    df['is_quote'] = df.apply(is_quote, axis=1)
    df['is_reply'] = df.apply(is_reply, axis=1)
    
    logger.info("Features de tweet créées: is_retweet, is_quote, is_reply")
    
    return df


def create_source_features(df: pd.DataFrame, source_column: str = 'source') -> pd.DataFrame:
    """
    Crée des features liées à la source du tweet
    
    Args:
        df: DataFrame source
        source_column: Nom de la colonne contenant la source
        
    Returns:
        DataFrame avec les nouvelles features
    """
    logger.info("Création des features de source")
    
    df = df.copy()
    
    if source_column in df.columns:
        df['source_type'] = df[source_column].apply(extract_source_type)
    
    logger.info("Features de source créées: source_type")
    
    return df


def create_all_features(df: pd.DataFrame, text_column: str = 'full_text') -> pd.DataFrame:
    """
    Crée toutes les features personnalisées
    
    Args:
        df: DataFrame source
        text_column: Nom de la colonne contenant le texte
        
    Returns:
        DataFrame avec toutes les features
    """
    logger.info("Création de toutes les features")
    
    df = create_text_features(df, text_column)
    df = create_tweet_features(df)
    df = create_source_features(df)
    
    logger.info("Toutes les features ont été créées")
    
    return df


if __name__ == "__main__":
    # Exemple d'utilisation
    logging.basicConfig(level=logging.INFO)
    
    # Exemple de texte
    sample_text = "Check out this amazing #Python tutorial! 🐍 @user1 http://example.com"
    
    print(f"Texte: {sample_text}")
    print(f"Hashtags: {count_hashtags(sample_text)}")
    print(f"Mentions: {count_mentions(sample_text)}")
    print(f"URLs: {count_urls(sample_text)}")
    print(f"Emojis: {count_emojis(sample_text)}")
    print(f"Longueur: {get_text_length(sample_text)}")
    print(f"Mots: {get_word_count(sample_text)}")
