"""
Metrics calculation utilities for model evaluation.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from typing import Dict, List, Tuple
import json
import os


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict:
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional, for AUC calculation)

    Returns:
        Dictionary containing all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

    # Add AUC if probabilities are provided
    if y_prob is not None:
        try:
            metrics['auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except:
            metrics['auc_ovr'] = 0.0

    return metrics


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str] = None) -> np.ndarray:
    """
    Calculate confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names (optional)

    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                             target_names: List[str] = None) -> str:
    """
    Generate detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Class names

    Returns:
        Classification report as string
    """
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


def save_metrics(metrics: Dict, save_path: str):
    """
    Save metrics to JSON file.

    Args:
        metrics: Dictionary of metrics
        save_path: Path to save the metrics
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Convert numpy types to native Python types for JSON serialization
    metrics_serializable = {}
    for key, value in metrics.items():
        if isinstance(value, (np.int64, np.int32)):
            metrics_serializable[key] = int(value)
        elif isinstance(value, (np.float64, np.float32)):
            metrics_serializable[key] = float(value)
        else:
            metrics_serializable[key] = value

    with open(save_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=4)

    print(f"Metrics saved to {save_path}")


def print_metrics(metrics: Dict, model_name: str = "Model"):
    """
    Pretty print metrics to console.

    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
    """
    print(f"\n{'='*60}")
    print(f"{model_name} Performance Metrics")
    print(f"{'='*60}")

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:.<30} {value:.4f}")
        else:
            print(f"{key:.<30} {value}")

    print(f"{'='*60}\n")


def calculate_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                class_names: List[str] = None) -> Dict:
    """
    Calculate per-class precision, recall, and F1 scores.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes

    Returns:
        Dictionary with per-class metrics
    """
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    per_class_metrics = {}
    num_classes = len(precision)

    for i in range(num_classes):
        class_name = class_names[i] if class_names else f"Class_{i}"
        per_class_metrics[class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i])
        }

    return per_class_metrics
