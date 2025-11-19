#!/usr/bin/env python3
"""
Script pour générer toutes les soumissions Kaggle automatiquement.
Utilise les modèles entraînés dans le notebook 04_advanced_models.ipynb

Usage:
    python generate_submissions.py --models all
    python generate_submissions.py --models ensemble
    python generate_submissions.py --models camembert
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import os
import sys
from pathlib import Path

# Ajouter src au path
sys.path.append(str(Path(__file__).parent))

from src.data_loader import load_jsonl, save_submission
from src.preprocessing import extract_full_text


def aggregate_predictions_by_user(df_original, predictions_per_tweet, method='mean'):
    """
    Agrège les prédictions au niveau utilisateur.
    
    Args:
        df_original: DataFrame original avec challenge_id
        predictions_per_tweet: array de prédictions (probas ou labels) par tweet
        method: 'mean' (moyenne des probas) ou 'majority' (vote majoritaire)
    
    Returns:
        user_predictions: prédictions finales par utilisateur
        user_ids: liste des challenge_ids
    """
    df_pred = df_original.copy()
    df_pred['prediction'] = predictions_per_tweet
    
    if method == 'mean':
        # Moyenne des probabilités par utilisateur
        user_agg = df_pred.groupby('challenge_id')['prediction'].mean()
        user_predictions = (user_agg >= 0.5).astype(int)
    else:  # majority
        # Vote majoritaire
        user_agg = df_pred.groupby('challenge_id')['prediction'].apply(
            lambda x: 1 if x.mean() >= 0.5 else 0
        )
        user_predictions = user_agg
    
    return user_predictions.values, user_agg.index.values


def load_and_prepare_data():
    """Charge et prépare les données."""
    print("📥 Chargement des données...")
    
    df_test = load_jsonl('data/kaggle_test.jsonl')
    df_test['full_text'] = df_test.apply(extract_full_text, axis=1)
    
    # Agréger par utilisateur
    test_grouped = df_test.groupby('challenge_id').agg({
        'full_text': lambda x: ' '.join(x.astype(str))
    }).reset_index()
    
    X_test = test_grouped['full_text']
    test_ids = test_grouped['challenge_id']
    
    print(f"✓ {len(X_test)} utilisateurs dans le test set")
    
    return df_test, X_test, test_ids, test_grouped


def generate_ensemble_submissions(df_test, test_ids_grouped):
    """Génère les soumissions d'ensemble à partir des modèles sauvegardés."""
    print("\n🎭 Génération des ensembles...")
    
    models_dir = Path('models')
    predictions = []
    model_names = []
    
    # Charger les modèles disponibles
    model_files = {
        'logistic_regression': 'logistic_regression_advanced.joblib',
        'svm': 'svm_advanced.joblib',
        'random_forest': 'random_forest_advanced.joblib',
        'lightgbm': 'lightgbm_advanced.joblib'
    }
    
    for name, filename in model_files.items():
        filepath = models_dir / filename
        if filepath.exists():
            print(f"  ✓ Chargement: {name}")
            model = joblib.load(filepath)
            
            # Faire les prédictions
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                scores = model.decision_function(X_test)
                proba = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                proba = model.predict(X_test)
            
            predictions.append(proba)
            model_names.append(name)
    
    if len(predictions) == 0:
        print("  ❌ Aucun modèle trouvé. Entraînez d'abord les modèles.")
        return
    
    predictions = np.array(predictions)
    print(f"  ✓ {len(predictions)} modèles chargés")
    
    # Ensemble vote
    ensemble_vote = (predictions >= 0.5).sum(axis=0) >= len(predictions) / 2
    ensemble_vote = ensemble_vote.astype(int)
    
    user_to_vote = dict(zip(test_ids_grouped, ensemble_vote))
    test_tweet_votes = df_test['challenge_id'].map(user_to_vote).values
    ensemble_vote_final, ensemble_vote_ids = aggregate_predictions_by_user(
        df_test, test_tweet_votes, method='majority'
    )
    
    # Ensemble mean
    ensemble_mean_proba = predictions.mean(axis=0)
    user_to_mean = dict(zip(test_ids_grouped, ensemble_mean_proba))
    test_tweet_mean = df_test['challenge_id'].map(user_to_mean).values
    ensemble_mean_final, ensemble_mean_ids = aggregate_predictions_by_user(
        df_test, test_tweet_mean, method='mean'
    )
    
    # Sauvegarder
    save_submission(ensemble_vote_ids, ensemble_vote_final, 
                   'submissions/ensemble_vote_submission.csv')
    save_submission(ensemble_mean_ids, ensemble_mean_final,
                   'submissions/ensemble_mean_submission.csv')
    
    print("  ✓ ensemble_vote_submission.csv")
    print("  ✓ ensemble_mean_submission.csv")


def generate_camembert_submission():
    """Génère la soumission CamemBERT si le modèle est disponible."""
    print("\n🤖 Génération de la soumission CamemBERT...")
    
    model_dir = Path('models/camembert_influencer_classifier')
    
    if not model_dir.exists():
        print("  ❌ Modèle CamemBERT non trouvé")
        print("     Entraînez d'abord le modèle dans le notebook avec USE_SAMPLE=False")
        return
    
    try:
        import torch
        from transformers import CamembertTokenizer, CamembertForSequenceClassification
        from datasets import Dataset
        
        # Charger le modèle
        tokenizer = CamembertTokenizer.from_pretrained(model_dir)
        model = CamembertForSequenceClassification.from_pretrained(model_dir)
        
        # Device
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        print(f"  ✓ Modèle chargé sur {device}")
        
        # Charger données
        df_test, X_test, test_ids, _ = load_and_prepare_data()
        
        # Tokenization
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=128)
        
        test_dataset = Dataset.from_dict({"text": X_test.tolist()})
        tokenized_test = test_dataset.map(tokenize_function, batched=True, num_proc=4)
        
        # Prédictions
        print("  → Génération des prédictions...")
        from transformers import Trainer, TrainingArguments
        
        args = TrainingArguments(
            output_dir="./tmp",
            per_device_eval_batch_size=64,
            report_to="none"
        )
        
        trainer = Trainer(model=model, args=args)
        predictions_output = trainer.predict(tokenized_test)
        predictions = predictions_output.predictions.argmax(-1)
        
        # Sauvegarder
        save_submission(test_ids, predictions, 'submissions/camembert_submission.csv')
        print("  ✓ camembert_submission.csv")
        
    except Exception as e:
        print(f"  ❌ Erreur: {e}")


def main():
    parser = argparse.ArgumentParser(description='Générer les soumissions Kaggle')
    parser.add_argument(
        '--models',
        choices=['all', 'ensemble', 'camembert'],
        default='all',
        help='Quels modèles générer'
    )
    
    args = parser.parse_args()
    
    print("🚀 Génération des Soumissions Kaggle\n")
    print("=" * 60)
    
    # Créer le dossier submissions s'il n'existe pas
    Path('submissions').mkdir(exist_ok=True)
    
    # Charger les données
    df_test, X_test, test_ids_grouped, test_grouped = load_and_prepare_data()
    
    # Définir X_test comme variable globale pour les fonctions
    globals()['X_test'] = X_test
    globals()['df_test'] = df_test
    
    # Générer les soumissions demandées
    if args.models in ['all', 'ensemble']:
        generate_ensemble_submissions(df_test, test_ids_grouped)
    
    if args.models in ['all', 'camembert']:
        generate_camembert_submission()
    
    print("\n" + "=" * 60)
    print("✅ Génération terminée!")
    print("\n📤 Fichiers disponibles dans submissions/:")
    
    submissions_dir = Path('submissions')
    for f in sorted(submissions_dir.glob('*_submission.csv')):
        size = f.stat().st_size
        print(f"   • {f.name} ({size} bytes)")
    
    print("\n🎯 Recommandation: Soumettez d'abord les ensembles!")


if __name__ == '__main__':
    main()
