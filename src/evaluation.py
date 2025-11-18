"""
Module d'évaluation des modèles
Ce module contient les fonctions pour évaluer les performances des modèles
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


def evaluate_model(y_true: np.ndarray, 
                   y_pred: np.ndarray, 
                   y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Évalue les performances d'un modèle
    
    Args:
        y_true: Vraies étiquettes
        y_pred: Prédictions
        y_pred_proba: Probabilités prédites (optionnel)
        
    Returns:
        Dictionnaire avec les métriques
    """
    logger.info("Évaluation du modèle")
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1': f1_score(y_true, y_pred, average='binary')
    }
    
    # Ajouter l'AUC-ROC si les probabilités sont fournies
    if y_pred_proba is not None:
        # Pour les probabilités binaires, prendre la probabilité de la classe 1
        if y_pred_proba.ndim > 1:
            y_pred_proba = y_pred_proba[:, 1]
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    
    # Logger les résultats
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1-Score: {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return metrics


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Affiche le rapport de classification détaillé
    
    Args:
        y_true: Vraies étiquettes
        y_pred: Prédictions
    """
    target_names = ['Observer (0)', 'Influencer (1)']
    report = classification_report(y_true, y_pred, target_names=target_names)
    
    print("\n=== Classification Report ===")
    print(report)


def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         save_path: Optional[str] = None,
                         figsize: tuple = (8, 6)) -> None:
    """
    Crée et affiche une matrice de confusion
    
    Args:
        y_true: Vraies étiquettes
        y_pred: Prédictions
        save_path: Chemin pour sauvegarder la figure (optionnel)
        figsize: Taille de la figure
    """
    logger.info("Création de la matrice de confusion")
    
    # Calculer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    
    # Créer la figure
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Observer', 'Influencer'],
                yticklabels=['Observer', 'Influencer'])
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.title('Matrice de Confusion')
    
    # Sauvegarder si un chemin est fourni
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Matrice de confusion sauvegardée dans {save_path}")
    
    plt.show()


def plot_roc_curve(y_true: np.ndarray, 
                   y_pred_proba: np.ndarray,
                   save_path: Optional[str] = None,
                   figsize: tuple = (8, 6)) -> None:
    """
    Crée et affiche la courbe ROC
    
    Args:
        y_true: Vraies étiquettes
        y_pred_proba: Probabilités prédites
        save_path: Chemin pour sauvegarder la figure (optionnel)
        figsize: Taille de la figure
    """
    logger.info("Création de la courbe ROC")
    
    # Calculer la courbe ROC
    if y_pred_proba.ndim > 1:
        y_pred_proba = y_pred_proba[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    # Créer la figure
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title('Courbe ROC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Sauvegarder si un chemin est fourni
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Courbe ROC sauvegardée dans {save_path}")
    
    plt.show()


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Compare les résultats de plusieurs modèles
    
    Args:
        results: Dictionnaire {nom_modele: {metrique: valeur}}
        
    Returns:
        DataFrame avec la comparaison
    """
    logger.info("Comparaison des modèles")
    
    df = pd.DataFrame(results).T
    df = df.sort_values('accuracy', ascending=False)
    
    print("\n=== Comparaison des Modèles ===")
    print(df.to_string())
    
    return df


def save_evaluation_report(metrics: Dict[str, float],
                           model_name: str,
                           save_path: str = 'reports/evaluation.txt') -> None:
    """
    Sauvegarde un rapport d'évaluation dans un fichier texte
    
    Args:
        metrics: Métriques du modèle
        model_name: Nom du modèle
        save_path: Chemin du fichier de sortie
    """
    logger.info(f"Sauvegarde du rapport d'évaluation dans {save_path}")
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'a') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Modèle: {model_name}\n")
        f.write(f"{'='*50}\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write(f"{'='*50}\n\n")
    
    logger.info("Rapport sauvegardé avec succès")


def plot_feature_importance(model, 
                           feature_names: list,
                           top_n: int = 20,
                           save_path: Optional[str] = None,
                           figsize: tuple = (10, 8)) -> None:
    """
    Affiche l'importance des features pour les modèles qui le supportent
    
    Args:
        model: Modèle entraîné (doit avoir feature_importances_)
        feature_names: Noms des features
        top_n: Nombre de features à afficher
        save_path: Chemin pour sauvegarder la figure (optionnel)
        figsize: Taille de la figure
    """
    logger.info("Création du graphique d'importance des features")
    
    # Récupérer les importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        logger.warning("Le modèle ne supporte pas l'importance des features")
        return
    
    # Créer un DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Trier et prendre les top_n
    feature_importance = feature_importance.sort_values('importance', ascending=False).head(top_n)
    
    # Créer la figure
    plt.figure(figsize=figsize)
    plt.barh(range(len(feature_importance)), feature_importance['importance'])
    plt.yticks(range(len(feature_importance)), feature_importance['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Features les plus importantes')
    plt.gca().invert_yaxis()
    
    # Sauvegarder si un chemin est fourni
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Graphique d'importance sauvegardé dans {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Exemple d'utilisation
    logging.basicConfig(level=logging.INFO)
    
    # Données d'exemple
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1, 1, 0])
    y_pred_proba = np.array([0.2, 0.8, 0.4, 0.3, 0.9, 0.1, 0.85, 0.95, 0.6, 0.15])
    
    # Évaluer
    metrics = evaluate_model(y_true, y_pred, y_pred_proba)
    print(f"\nMétriques: {metrics}")
    
    # Rapport de classification
    print_classification_report(y_true, y_pred)
