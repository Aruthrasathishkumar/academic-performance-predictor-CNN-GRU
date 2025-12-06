"""
Train baseline models (SVM and Random Forest) for performance comparison.
Uses aggregated features from sequences for traditional ML models.
"""

import numpy as np
import argparse
import pickle
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

from dataset import load_processed_data, split_data
from utils.config import (
    BASELINE_CONFIG, TRAINING_CONFIG, CHECKPOINTS_DIR, METRICS_DIR
)
from utils.metrics import (
    calculate_metrics, print_metrics, save_metrics,
    get_classification_report, get_confusion_matrix
)
from utils.logging_utils import get_default_logger

logger = get_default_logger('train_baselines')


def aggregate_sequence_features(sequences: np.ndarray) -> np.ndarray:
    """
    Aggregate sequence features into static features for traditional ML.

    Args:
        sequences: Input sequences of shape (num_students, seq_len, num_features)

    Returns:
        Aggregated features of shape (num_students, num_aggregated_features)
    """
    num_students, seq_len, num_features = sequences.shape

    # Statistical aggregations across the sequence dimension
    features_list = []

    # Mean, std, min, max for each feature
    features_list.append(np.mean(sequences, axis=1))  # (num_students, num_features)
    features_list.append(np.std(sequences, axis=1))
    features_list.append(np.min(sequences, axis=1))
    features_list.append(np.max(sequences, axis=1))

    # Median
    features_list.append(np.median(sequences, axis=1))

    # First and last values (temporal information)
    features_list.append(sequences[:, 0, :])  # First time step
    features_list.append(sequences[:, -1, :])  # Last time step

    # Trend (difference between last and first)
    features_list.append(sequences[:, -1, :] - sequences[:, 0, :])

    # Concatenate all features
    aggregated_features = np.concatenate(features_list, axis=1)

    logger.info(f"Aggregated features shape: {aggregated_features.shape}")

    return aggregated_features


def train_svm(X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> SVC:
    """
    Train SVM classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels

    Returns:
        Trained SVM model
    """
    logger.info("Training SVM classifier...")

    svm_config = BASELINE_CONFIG['svm']
    svm = SVC(**svm_config, probability=True, verbose=False)

    start_time = time.time()
    svm.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Evaluate
    y_train_pred = svm.predict(X_train)
    y_val_pred = svm.predict(X_val)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)

    logger.info(f"SVM Training completed in {train_time:.2f} seconds")
    logger.info(f"SVM Train Accuracy: {train_acc:.4f}")
    logger.info(f"SVM Validation Accuracy: {val_acc:.4f}")

    return svm


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray) -> RandomForestClassifier:
    """
    Train Random Forest classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels

    Returns:
        Trained Random Forest model
    """
    logger.info("Training Random Forest classifier...")

    rf_config = BASELINE_CONFIG['random_forest']
    rf = RandomForestClassifier(**rf_config, verbose=0)

    start_time = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Evaluate
    y_train_pred = rf.predict(X_train)
    y_val_pred = rf.predict(X_val)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)

    logger.info(f"Random Forest Training completed in {train_time:.2f} seconds")
    logger.info(f"Random Forest Train Accuracy: {train_acc:.4f}")
    logger.info(f"Random Forest Validation Accuracy: {val_acc:.4f}")

    return rf


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                  model_name: str) -> dict:
    """
    Evaluate a trained model.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model

    Returns:
        Dictionary of metrics
    """
    logger.info(f"Evaluating {model_name}...")

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_prob)

    # Print results
    print_metrics(metrics, model_name)

    # Classification report
    report = get_classification_report(y_test, y_pred)
    logger.info(f"\nClassification Report for {model_name}:\n{report}")

    # Confusion matrix
    cm = get_confusion_matrix(y_test, y_pred)
    logger.info(f"\nConfusion Matrix for {model_name}:\n{cm}")

    return metrics


def save_model(model, model_name: str):
    """
    Save trained model to disk.

    Args:
        model: Trained model
        model_name: Name for the saved model
    """
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    filepath = os.path.join(CHECKPOINTS_DIR, f'{model_name}.pkl')

    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

    logger.info(f"Model saved to {filepath}")


def main():
    """Main function for training baseline models."""
    parser = argparse.ArgumentParser(description='Train baseline models (SVM, Random Forest)')
    parser.add_argument('--data_prefix', type=str, default='train',
                       help='Prefix for preprocessed data files')
    parser.add_argument('--models', nargs='+', default=['svm', 'rf'],
                       choices=['svm', 'rf', 'both'],
                       help='Models to train: svm, rf, or both')
    parser.add_argument('--save_models', action='store_true',
                       help='Save trained models to disk')

    args = parser.parse_args()

    # Set random seed
    np.random.seed(TRAINING_CONFIG['random_seed'])

    # Load preprocessed data
    logger.info("Loading preprocessed data...")
    sequences, labels, student_ids = load_processed_data(args.data_prefix)

    # Split data
    logger.info("Splitting data into train/val/test sets...")
    train_data, val_data, test_data = split_data(
        sequences, labels, student_ids,
        val_split=TRAINING_CONFIG['validation_split'],
        test_split=TRAINING_CONFIG['test_split'],
        random_seed=TRAINING_CONFIG['random_seed']
    )

    # Unpack
    X_train_seq, y_train, train_ids = train_data
    X_val_seq, y_val, val_ids = val_data
    X_test_seq, y_test, test_ids = test_data

    logger.info(f"Train samples: {len(y_train)}")
    logger.info(f"Val samples: {len(y_val)}")
    logger.info(f"Test samples: {len(y_test)}")

    # Aggregate features for traditional ML
    logger.info("Aggregating sequence features...")
    X_train = aggregate_sequence_features(X_train_seq)
    X_val = aggregate_sequence_features(X_val_seq)
    X_test = aggregate_sequence_features(X_test_seq)

    results = {}

    # Determine which models to train
    if 'both' in args.models:
        train_models = ['svm', 'rf']
    else:
        train_models = args.models

    # Train SVM
    if 'svm' in train_models:
        svm_model = train_svm(X_train, y_train, X_val, y_val)
        svm_metrics = evaluate_model(svm_model, X_test, y_test, 'SVM')
        results['svm'] = svm_metrics

        if args.save_models:
            save_model(svm_model, 'svm_baseline')

        # Save metrics
        save_metrics(svm_metrics, os.path.join(METRICS_DIR, 'svm_metrics.json'))

    # Train Random Forest
    if 'rf' in train_models:
        rf_model = train_random_forest(X_train, y_train, X_val, y_val)
        rf_metrics = evaluate_model(rf_model, X_test, y_test, 'Random Forest')
        results['random_forest'] = rf_metrics

        if args.save_models:
            save_model(rf_model, 'random_forest_baseline')

        # Save metrics
        save_metrics(rf_metrics, os.path.join(METRICS_DIR, 'random_forest_metrics.json'))

    # Summary comparison
    logger.info("\n" + "="*60)
    logger.info("BASELINE MODELS COMPARISON")
    logger.info("="*60)

    for model_name, metrics in results.items():
        logger.info(f"\n{model_name.upper()}:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        logger.info(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")

    logger.info("\n" + "="*60)
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
