# 🏆 TODO - Stratégie pour Gagner le Kaggle "Influencer or Observer"

> **Objectif**: Classifier les tweets français comme "Influencer" (1) ou "Observer" (0)
> **Métrique**: Accuracy
> **Deadline**: 3 jours restants

---

## 📊 Analyse du Projet Actuel

### Données
| Ensemble | Tweets | Utilisateurs | Features |
|----------|--------|--------------|----------|
| Train | 154,914 | 38,560 | 194 |
| Test | 103,380 | 25,890 | - |

### Notebooks Existants
| Notebook | Description | Status |
|----------|-------------|--------|
| `Data/baseline.ipynb` | TF-IDF + LR + LightGBM + Ensemble | ✅ Amélioré |
| `finetune_camembert.ipynb` | Fine-tuning CamemBERT end-to-end | ✅ Prêt |
| `hierarchical.ipynb` | Approche hiérarchique par utilisateur | 🔄 À compléter |
| `preprocessing.ipynb` | Extraction embeddings CamemBERT | ✅ Prêt |
| `model.ipynb` | XGBoost avec embeddings | ✅ Prêt |

### Labs Analysés
| Lab | Techniques Clés | Application |
|-----|-----------------|-------------|
| **Lab4** | LSTM, Embeddings, Dropout, Régularisation | Text classification |
| **Lab5** | Transformers, Transfer Learning, Positional Encoding | CamemBERT fine-tuning |
| **Lab6** | Feature Engineering, t-SNE, Représentations visuelles | Features combinées |
| **Lab8** | Adam vs SGD, Hyperparameter tuning, Optuna | Optimisation |

---

## 🎯 Plan d'Action Détaillé

### PHASE 1: Quick Wins (2-4 heures) ⚡

#### 1.1 Améliorer baseline.ipynb
- [x] ~~Ajouter TF-IDF avec trigrams~~
- [x] ~~Ajouter LightGBM~~
- [x] ~~Ajouter features structurées~~
- [x] ~~Ajouter Ensemble Voting~~
- [ ] **Exécuter le notebook et soumettre les résultats**

#### 1.2 Optimiser les hyperparamètres TF-IDF
```python
# À tester dans baseline.ipynb
param_grid = {
    'max_features': [5000, 10000, 20000],
    'ngram_range': [(1,2), (1,3), (2,3)],
    'max_df': [0.7, 0.8, 0.9],
    'min_df': [1, 2, 3],
    'sublinear_tf': [True, False],
    'analyzer': ['word', 'char_wb']  # Character n-grams!
}
```

---

### PHASE 2: Feature Engineering Avancé (3-5 heures) 🔧 ✅ COMPLÉTÉ

> **Nouveau notebook**: `feature_engineering.ipynb` ✅ CRÉÉ

#### 2.1 Features Textuelles Discriminantes ✅

```python
# Features qui distinguent Influenceurs vs Observateurs
features_influencer = {
    # Influenceurs ont tendance à:
    'call_to_action': r'suivez|abonnez|cliquez|like|partage|rt|retweet',
    'self_promotion': r'nouveau|nouvelle|vidéo|article|podcast|live',
    'hashtag_heavy': lambda x: len(re.findall(r'#\w+', x)) > 3,
    'emoji_heavy': lambda x: emoji_count(x) > 5,
    'longer_tweets': lambda x: len(x) > 200,
    
    # Observateurs ont tendance à:
    'reply_heavy': lambda x: x.startswith('@'),
    'more_mentions': lambda x: len(re.findall(r'@\w+', x)) > 2,
    'quote_retweet': r'RT @|QT:',
    'shorter_tweets': lambda x: len(x) < 100,
}
```

#### 2.2 Features Structurées du Tweet
| Feature | Description | Importance |
|---------|-------------|------------|
| `source` | Device (iPhone, Android, Web, Bot?) | ⭐⭐⭐ |
| `retweet_count` | Nombre de RT | ⭐⭐⭐ |
| `favorite_count` | Nombre de likes | ⭐⭐⭐ |
| `is_reply` | Est une réponse | ⭐⭐ |
| `is_quote_status` | Est un QT | ⭐⭐ |
| `user.statuses_count` | Nombre total de tweets | ⭐⭐⭐ |
| `user.favourites_count` | Nombre de likes donnés | ⭐⭐ |

#### 2.3 Features NLP Avancées (Lab4 inspiré) ✅
- [x] **Sentiment Analysis**: Utiliser un modèle pré-entraîné français
- [x] **Named Entity Recognition**: Détecter les mentions de marques, personnes
- [ ] **Topic Modeling**: LDA pour extraire les thèmes (optionnel)

```python
# Ajouter dans feature_engineering.ipynb
from transformers import pipeline

# Sentiment (Lab4 concept)
sentiment_pipeline = pipeline("sentiment-analysis", 
                              model="nlptown/bert-base-multilingual-uncased-sentiment")

# NER français
ner_pipeline = pipeline("ner", model="Jean-Baptiste/camembert-ner")
```

---

### PHASE 3: Transfer Learning avec CamemBERT (4-8 heures) 🤖

> **Notebook existant**: `finetune_camembert.ipynb`

#### 3.1 Configuration Optimale (Inspiré Lab5)

```python
# Hyperparamètres recommandés pour CamemBERT
training_args = TrainingArguments(
    # Epochs: 2-4 (plus = overfitting)
    num_train_epochs=3,
    
    # Learning rate: 2e-5 optimal pour BERT
    learning_rate=2e-5,
    
    # Batch size: 16 ou 32
    per_device_train_batch_size=16,
    
    # Warmup: 6-10% des steps
    warmup_ratio=0.1,
    
    # Weight decay (régularisation Lab4)
    weight_decay=0.01,
    
    # Gradient clipping
    max_grad_norm=1.0,
    
    # Mixed precision (2x plus rapide)
    fp16=True,
)
```

#### 3.2 Améliorations à implémenter

- [ ] **Layer-wise Learning Rate Decay (LLRD)**
  ```python
  # Couches profondes = LR plus petit
  # Inspiré de Lab5 (Transformer architecture)
  layer_decay = 0.95
  lr_layer_n = lr * (layer_decay ** (12 - n))
  ```

- [ ] **Multi-Sample Dropout** (Lab4 Dropout concept)
  ```python
  class MultiSampleDropout(nn.Module):
      def __init__(self, dropout_probs=[0.1, 0.2, 0.3, 0.4, 0.5]):
          self.dropouts = [nn.Dropout(p) for p in dropout_probs]
      
      def forward(self, x):
          # Moyenne des prédictions avec différents dropouts
          return torch.stack([d(x) for d in self.dropouts]).mean(0)
  ```

- [ ] **Pooling Strategy**: Tester différentes méthodes
  - CLS token (défaut)
  - Mean pooling
  - Max pooling
  - Attention pooling

---

### PHASE 4: Modèles Alternatifs (4-6 heures) 🔬

#### 4.1 FlauBERT (pour diversité d'ensemble)

> **Nouveau notebook**: `finetune_flaubert.ipynb`

```python
from transformers import FlaubertTokenizer, FlaubertForSequenceClassification

model_name = "flaubert/flaubert_base_cased"  # ou flaubert_large_cased
tokenizer = FlaubertTokenizer.from_pretrained(model_name)
model = FlaubertForSequenceClassification.from_pretrained(model_name, num_labels=2)
```

#### 4.2 LSTM avec CamemBERT Embeddings (Lab4 + Lab5)

> **Nouveau notebook**: `lstm_classifier.ipynb`

```python
# Inspiré de Lab4 - LSTM pour text classification
class LSTMClassifier(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, 2)  # bidirectional = *2
        self.dropout = nn.Dropout(dropout)  # Lab4 regularization
    
    def forward(self, embeddings):
        # embeddings: [batch, seq_len, 768] from CamemBERT
        lstm_out, (h_n, c_n) = self.lstm(embeddings)
        # Concat forward and backward hidden states
        hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        out = self.dropout(hidden)
        return self.fc(out)
```

#### 4.3 Approche Hiérarchique (Compléter hierarchical.ipynb)

```python
# Agréger les tweets par utilisateur (moyenne des embeddings)
# Puis classifier au niveau utilisateur

class HierarchicalClassifier(nn.Module):
    def __init__(self, emb_dim=768, hidden_dim=256):
        super().__init__()
        # Attention sur les tweets de l'utilisateur
        self.attention = nn.MultiheadAttention(emb_dim, num_heads=8)
        self.fc = nn.Linear(emb_dim, 2)
    
    def forward(self, user_tweets):
        # user_tweets: [num_tweets, emb_dim]
        attn_out, _ = self.attention(user_tweets, user_tweets, user_tweets)
        # Mean pooling après attention
        user_emb = attn_out.mean(dim=0)
        return self.fc(user_emb)
```

---

### PHASE 5: Ensemble et Stacking (3-5 heures) 🎲

> **Nouveau notebook**: `ensemble_stacking.ipynb`

#### 5.1 Générer les OOF Predictions

```python
# Pour chaque modèle, générer des prédictions Out-of-Fold
models = {
    'camembert': CamemBERTClassifier(),
    'flaubert': FlauBERTClassifier(),
    'lgbm_tfidf': LGBMPipeline(),
    'xgb_features': XGBClassifier(),
    'lstm_camembert': LSTMClassifier(),
}

oof_predictions = {}
test_predictions = {}

for name, model in models.items():
    oof, test = get_oof_predictions(model, X_train, y_train, X_test, n_splits=5)
    oof_predictions[name] = oof
    test_predictions[name] = test
```

#### 5.2 Optimiser les Poids avec Optuna (Lab8)

```python
import optuna

def optimize_ensemble_weights(trial, oof_preds, y_true):
    weights = {}
    for name in oof_preds.keys():
        weights[name] = trial.suggest_float(f'w_{name}', 0, 1)
    
    # Normaliser
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}
    
    # Prédiction ensemble
    ensemble_pred = sum(oof_preds[k] * w for k, w in weights.items())
    y_pred = (ensemble_pred >= 0.5).astype(int)
    
    return accuracy_score(y_true, y_pred)

study = optuna.create_study(direction='maximize')
study.optimize(lambda t: optimize_ensemble_weights(t, oof_predictions, y_train), 
               n_trials=100)

print(f"Best weights: {study.best_params}")
print(f"Best accuracy: {study.best_value}")
```

#### 5.3 Stacking avec Meta-Learner

```python
from sklearn.ensemble import StackingClassifier

# Niveau 1: Modèles de base
base_estimators = [
    ('camembert_probs', camembert_model),
    ('flaubert_probs', flaubert_model),
    ('lgbm_probs', lgbm_model),
    ('structured_features', structured_model),
]

# Niveau 2: Meta-learner
stacking_clf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(C=0.1),
    cv=5,
    stack_method='predict_proba',
    passthrough=True  # Inclure features originales
)
```

---

### PHASE 6: Techniques Avancées (2-4 heures) 🚀

#### 6.1 Pseudo-Labeling

```python
# Utiliser les prédictions confiantes sur test pour augmenter train
def pseudo_labeling(model, X_train, y_train, X_test, threshold=0.95):
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    
    confident_mask = probs.max(axis=1) >= threshold
    pseudo_labels = probs.argmax(axis=1)[confident_mask]
    pseudo_X = X_test[confident_mask]
    
    # Ré-entraîner avec pseudo-labels
    X_augmented = np.vstack([X_train, pseudo_X])
    y_augmented = np.hstack([y_train, pseudo_labels])
    
    model.fit(X_augmented, y_augmented)
    return model
```

#### 6.2 Test-Time Augmentation (TTA)

```python
def predict_with_tta(model, texts, n_aug=5):
    predictions = []
    
    for text in texts:
        preds = [model.predict_proba([text])[0]]
        
        # Augmentations
        for _ in range(n_aug):
            aug_text = augment_text(text)  # Synonym replacement, etc.
            preds.append(model.predict_proba([aug_text])[0])
        
        # Moyenne
        predictions.append(np.mean(preds, axis=0))
    
    return np.array(predictions)
```

#### 6.3 Optimisation du Seuil de Décision

```python
from sklearn.metrics import precision_recall_curve

def find_optimal_threshold(y_true, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs[:, 1])
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    print(f"Optimal threshold: {best_threshold:.4f}")
    return best_threshold

# Appliquer sur les prédictions finales
optimal_thresh = find_optimal_threshold(y_val, val_probs)
final_predictions = (test_probs[:, 1] >= optimal_thresh).astype(int)
```

---

### PHASE 7: Data Augmentation (2-3 heures) 📈

> **Nouveau notebook**: `data_augmentation.ipynb`

#### 7.1 Techniques à implémenter

```python
import nlpaug.augmenter.word as naw

# 1. Back-translation (FR -> EN -> FR)
from transformers import MarianMTModel, MarianTokenizer

def back_translate(text, src='fr', pivot='en'):
    # fr -> en
    en_text = translate(text, f'{src}-{pivot}')
    # en -> fr
    fr_text = translate(en_text, f'{pivot}-{src}')
    return fr_text

# 2. Synonym Replacement avec CamemBERT
aug_contextual = naw.ContextualWordEmbsAug(
    model_path='camembert-base',
    action="substitute",
    aug_p=0.1
)

# 3. Random Insertion/Deletion (EDA)
aug_random = naw.RandomWordAug(action="delete", aug_p=0.1)
```

#### 7.2 Stratégie d'Augmentation

```python
# Augmenter SEULEMENT la classe minoritaire
# NE JAMAIS augmenter les données de validation!

def augment_minority_class(df, text_col, label_col, minority_label, n_aug=2):
    minority_df = df[df[label_col] == minority_label]
    augmented = []
    
    for _, row in minority_df.iterrows():
        for _ in range(n_aug):
            aug_text = aug_contextual.augment(row[text_col])
            new_row = row.copy()
            new_row[text_col] = aug_text
            augmented.append(new_row)
    
    return pd.concat([df, pd.DataFrame(augmented)])
```

---

## 📋 Checklist Finale

### Avant Soumission
- [ ] Cross-validation 5-fold sur tous les modèles
- [ ] Vérifier la distribution des prédictions (pas trop déséquilibrée)
- [ ] Vérifier le format du fichier de soumission (ID, Prediction)
- [ ] Tester plusieurs seuils de décision

### Soumissions Recommandées (par ordre de priorité)

| Priorité | Modèle | Accuracy Estimée | Temps |
|----------|--------|------------------|-------|
| 1️⃣ | CamemBERT fine-tuné | 85-88% | 4h GPU |
| 2️⃣ | Ensemble (CamemBERT + FlauBERT + LightGBM) | 87-90% | 8h GPU |
| 3️⃣ | LightGBM + TF-IDF + Features | 78-82% | 1h CPU |
| 4️⃣ | Baseline TF-IDF + LR | 72-75% | 10min CPU |

---

## 📚 Références Labs

### Lab4 - Text Processing & Regularization
- **Tokenization & Vocabulary**: Utilisé pour TF-IDF
- **Embeddings**: Concept appliqué avec CamemBERT
- **LSTM**: Implémentation possible avec embeddings
- **Dropout**: Utilisé dans tous les modèles deep learning
- **Weight Decay**: Paramètre C dans LogisticRegression, weight_decay dans Transformers

### Lab5 - Transformers & Transfer Learning
- **Positional Encoding**: Natif dans CamemBERT
- **Transformer Architecture**: CamemBERT/FlauBERT
- **Transfer Learning**: Fine-tuning vs From Scratch
- **Classification Head**: Remplacer la tête pour notre tâche

### Lab6 - Feature Engineering
- **Feature Extraction**: Features structurées des tweets
- **Representation Learning**: Embeddings CamemBERT
- **Visualization (t-SNE)**: Pour analyser les embeddings

### Lab8 - Optimization
- **Adam Optimizer**: Meilleur pour transformers
- **Learning Rate Tuning**: GridSearchCV, Optuna
- **Hyperparameter Search**: Grille de paramètres

---

## ⏱️ Timeline Suggérée (3 jours)

### Jour 1
- [ ] Matin: Exécuter baseline.ipynb, soumettre premiers résultats
- [ ] Après-midi: Feature engineering avancé
- [ ] Soir: Lancer fine-tuning CamemBERT (GPU)

### Jour 2
- [ ] Matin: Analyser résultats CamemBERT, ajuster hyperparamètres
- [ ] Après-midi: Fine-tuning FlauBERT pour diversité
- [ ] Soir: Commencer ensemble/stacking

### Jour 3
- [ ] Matin: Finaliser ensemble, optimiser poids
- [ ] Après-midi: Pseudo-labeling, TTA
- [ ] Soir: Soumissions finales, backup plans

---

## 🔗 Ressources Utiles

- [CamemBERT Paper](https://arxiv.org/abs/1911.03894)
- [FlauBERT Paper](https://arxiv.org/abs/1912.05372)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [Kaggle NLP Competition Tips](https://www.kaggle.com/competitions)
- [nlpaug Documentation](https://github.com/makcedward/nlpaug)
