# Description des Données

Ce dossier contient les données du challenge Kaggle "Influencer or Observer: Predicting Social Roles".

## Fichiers

### `train.jsonl`
- **Format**: JSON Lines (un objet JSON par ligne)
- **Taille**: 154,914 tweets
- **Utilisateurs**: 38,560 utilisateurs
- **Features**: ~194 colonnes
- **Target**: Colonne `label` (0 = Observer, 1 = Influencer)

### `kaggle_test.jsonl`
- **Format**: JSON Lines
- **Taille**: 103,380 tweets  
- **Utilisateurs**: 25,890 utilisateurs
- **Features**: ~194 colonnes (sans la colonne `label`)
- **ID**: Colonne `challenge_id` pour la soumission

## Structure des Données

Chaque ligne JSON représente un tweet avec la structure suivante :

### Champs principaux
- `text`: Texte du tweet (peut être tronqué)
- `extended_tweet.full_text`: Texte complet si le tweet est tronqué
- `source`: Source/plateforme utilisée pour tweeter
- `created_at`: Date de création du tweet
- `id_str`: ID unique du tweet
- `lang`: Langue détectée du tweet

### Champs utilisateur (préfixés par `user.`)
- `user.statuses_count`: Nombre de tweets de l'utilisateur
- `user.location`: Localisation de l'utilisateur
- `user.description`: Description du profil
- `user.verified`: Si l'utilisateur est vérifié

### Champs d'interaction
- `retweet_count`: Nombre de retweets
- `favorite_count`: Nombre de likes
- `in_reply_to_status_id`: ID du tweet auquel on répond (si réponse)
- `quoted_status`: Tweet cité (si citation)
- `retweeted_status`: Tweet retweeté (si retweet)

### Champs entités
- `entities.urls`: URLs dans le tweet
- `entities.hashtags`: Hashtags dans le tweet
- `entities.user_mentions`: Mentions d'utilisateurs
- `entities.media`: Médias attachés

### Champ cible (training set uniquement)
- `label`: 0 = Observer, 1 = Influencer
- `challenge_id`: ID unique pour la soumission

## Notes Importantes

1. **Texte complet**: Toujours vérifier `extended_tweet.full_text` pour obtenir le texte complet
2. **Valeurs manquantes**: De nombreux champs peuvent être `null` ou absents
3. **Données imbriquées**: Structure JSON imbriquée nécessitant `json_normalize()`
4. **Langue**: Les tweets sont principalement en français
5. **Contexte COVID-19**: Les tweets semblent être dans le contexte de la pandémie COVID-19

## Distribution des Classes (Training Set)

- Analysez la distribution avec le notebook `01_EDA.ipynb`
- Vérifiez si les classes sont équilibrées
- Considérez des techniques de rééquilibrage si nécessaire

## Exemples d'Utilisation

```python
import pandas as pd
from pandas import json_normalize

# Charger les données
train_data = pd.read_json('train.jsonl', lines=True)
train_data = json_normalize(train_data.to_dict(orient='records'))

# Accéder au texte complet
def extract_full_text(tweet):
    text = tweet['text']
    if not pd.isna(tweet.get('extended_tweet.full_text')):
        text = tweet['extended_tweet.full_text']
    return text

train_data['full_text'] = train_data.apply(extract_full_text, axis=1)
```

## Considérations pour la Modélisation

### Features textuelles potentielles
- Longueur du texte
- Nombre de hashtags, mentions, URLs
- Présence d'emojis
- Complexité du vocabulaire
- Sentiment

### Features utilisateur potentielles
- `user.statuses_count` (activité)
- `user.location` (géographie)
- Source du tweet (mobile vs web vs automation tools)

### Stratégies de preprocessing
- Nettoyage du texte (URLs, mentions)
- Tokenization
- Stopwords (français)
- TF-IDF ou embeddings
