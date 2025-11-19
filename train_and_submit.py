# Example: Script d'Entraînement et Soumission

"""
Script exemple montrant comment utiliser les modules src/
pour entraîner un modèle et créer une soumission Kaggle
"""

import logging
from pathlib import Path

# Imports des modules du projet
from src.data_loader import load_training_data, load_test_data, save_submission
from src.preprocessing import preprocess_dataframe
from src.feature_engineering import create_all_features
from src.models import LogisticRegressionModel, save_model
from src.evaluation import evaluate_model, print_classification_report
from src.utils import setup_logging, load_config, print_section

def main():
    """Fonction principale"""
    
    # Configuration du logging
    setup_logging(log_file='reports/training.log', level='INFO')
    logger = logging.getLogger(__name__)
    
    print_section("INFLUENCER OR OBSERVER - TRAINING PIPELINE")
    
    # ============================================================================
    # 1. CHARGEMENT DE LA CONFIGURATION
    # ============================================================================
    print_section("1. CONFIGURATION", char='-')
    config = load_config('config/config.json')
    logger.info("Configuration chargée")
    
    # ============================================================================
    # 2. CHARGEMENT DES DONNÉES
    # ============================================================================
    print_section("2. CHARGEMENT DES DONNÉES", char='-')
    
    X_train, y_train = load_training_data(config['paths']['train_file'])
    X_test = load_test_data(config['paths']['test_file'])
    
    print(f"Données d'entraînement: {X_train.shape}")
    print(f"Données de test: {X_test.shape}")
    
    # ============================================================================
    # 3. PREPROCESSING
    # ============================================================================
    print_section("3. PREPROCESSING", char='-')
    
    clean_params = {
        'remove_url': config['preprocessing']['remove_urls'],
        'remove_mention': config['preprocessing']['remove_mentions'],
        'remove_hashtag': config['preprocessing']['remove_hashtags'],
        'lowercase': config['preprocessing']['lowercase'],
    }
    
    # Prétraiter les données
    initial_train_size = len(X_train)
    X_train = preprocess_dataframe(X_train, clean_params=clean_params)
    X_test = preprocess_dataframe(X_test, clean_params=clean_params)
    
    # Filtrer y_train pour correspondre aux indices de X_train après preprocessing
    # (des lignes peuvent avoir été supprimées si le texte était trop court)
    if len(X_train) != initial_train_size:
        print(f"⚠️ {initial_train_size - len(X_train)} tweets supprimés (trop courts)")
        y_train = y_train.loc[X_train.index]
        print(f"✓ y_train ajusté: {len(y_train)} échantillons")
    
    print(f"✓ Données après preprocessing:")
    print(f"  - Train: {len(X_train)} échantillons")
    print(f"  - Test: {len(X_test)} échantillons")
    
    # ============================================================================
    # 4. FEATURE ENGINEERING (Optionnel)
    # ============================================================================
    print_section("4. FEATURE ENGINEERING", char='-')
    
    # Créer des features supplémentaires si nécessaire
    # X_train = create_all_features(X_train)
    # X_test = create_all_features(X_test)
    
    print("Feature engineering: Skipped (using text only)")
    
    # ============================================================================
    # 5. ENTRAÎNEMENT DU MODÈLE
    # ============================================================================
    print_section("5. ENTRAÎNEMENT", char='-')
    
    # Créer le modèle avec les paramètres de config
    model = LogisticRegressionModel(
        max_features=config['tfidf']['max_features'],
        max_df=config['tfidf']['max_df'],
        min_df=config['tfidf']['min_df'],
        ngram_range=tuple(config['tfidf']['ngram_range']),
        use_stopwords=config['tfidf']['use_stopwords'],
        stopwords_language=config['tfidf']['stopwords_language'],
        solver=config['logistic_regression']['solver'],
        C=config['logistic_regression']['C'],
        random_state=config['random_state']
    )
    
    # Validation croisée
    print("\nValidation croisée en cours...")
    cv_results = model.cross_validate(
        X_train['full_text'], 
        y_train,
        cv=config['validation']['cv_folds']
    )
    
    print(f"\n✓ Score moyen (CV): {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")
    print(f"  Min: {cv_results['min_score']:.4f}")
    print(f"  Max: {cv_results['max_score']:.4f}")
    
    # Entraînement final sur toutes les données
    print("\nEntraînement final sur toutes les données...")
    model.fit(X_train['full_text'], y_train)
    print("✓ Modèle entraîné")
    
    # ============================================================================
    # 6. PRÉDICTIONS
    # ============================================================================
    print_section("6. PRÉDICTIONS", char='-')
    
    # Prédictions sur le test set
    y_pred = model.predict(X_test['full_text'])
    
    print(f"Nombre de prédictions: {len(y_pred)}")
    print(f"Distribution des prédictions:")
    print(f"  - Classe 0 (Observer): {(y_pred == 0).sum()}")
    print(f"  - Classe 1 (Influencer): {(y_pred == 1).sum()}")
    
    # ============================================================================
    # 7. SAUVEGARDE
    # ============================================================================
    print_section("7. SAUVEGARDE", char='-')
    
    # Sauvegarder le modèle
    model_filename = f"models/logistic_regression_{cv_results['mean_score']:.4f}.joblib"
    save_model(model, model_filename)
    print(f"✓ Modèle sauvegardé: {model_filename}")
    
    # Créer la soumission
    submission_filename = f"submissions/logistic_regression_{cv_results['mean_score']:.4f}.csv"
    save_submission(y_pred, X_test['challenge_id'], submission_filename)
    print(f"✓ Soumission créée: {submission_filename}")
    
    # ============================================================================
    # FIN
    # ============================================================================
    print_section("TERMINÉ", char='=')
    print(f"\n📊 Résultats:")
    print(f"   - Accuracy (CV): {cv_results['mean_score']:.4f}")
    print(f"   - Modèle: {model_filename}")
    print(f"   - Soumission: {submission_filename}")
    print(f"\n🎯 Prêt pour soumission sur Kaggle!")
    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()
