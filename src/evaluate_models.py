"""
Evaluate and compare all models (CNN-GRU and baselines).
"""

import torch
import numpy as np
import pickle
import json
import os
import argparse
from tabulate import tabulate

from models.cnn_gru_model import CNNGRU
from dataset import load_processed_data, split_data
from train_baselines import aggregate_sequence_features
from utils.config import (
    CNN_GRU_CONFIG, TRAINING_CONFIG, CHECKPOINTS_DIR, METRICS_DIR
)
from utils.metrics import (
    calculate_metrics, get_classification_report, get_confusion_matrix,
    calculate_per_class_metrics
)
from utils.logging_utils import get_default_logger

logger = get_default_logger('evaluate_models')


def load_cnn_gru_model(checkpoint_path: str, device: torch.device) -> CNNGRU:
    """
    Load trained CNN-GRU model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded model
    """
    logger.info(f"Loading CNN-GRU model from {checkpoint_path}")

    model = CNNGRU(
        input_dim=CNN_GRU_CONFIG['input_dim'],
        cnn_channels=CNN_GRU_CONFIG['cnn_channels'],
        kernel_sizes=CNN_GRU_CONFIG['kernel_sizes'],
        gru_hidden_dim=CNN_GRU_CONFIG['gru_hidden_dim'],
        gru_num_layers=CNN_GRU_CONFIG['gru_num_layers'],
        num_classes=CNN_GRU_CONFIG['num_classes'],
        dropout=CNN_GRU_CONFIG['dropout'],
        bidirectional=CNN_GRU_CONFIG['bidirectional']
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info("CNN-GRU model loaded successfully")
    return model


def load_baseline_model(model_path: str):
    """
    Load trained baseline model from pickle file.

    Args:
        model_path: Path to pickle file

    Returns:
        Loaded model
    """
    logger.info(f"Loading baseline model from {model_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    logger.info("Baseline model loaded successfully")
    return model


def evaluate_cnn_gru(model, sequences: np.ndarray, labels: np.ndarray,
                     device: torch.device) -> tuple:
    """
    Evaluate CNN-GRU model.

    Args:
        model: Trained model
        sequences: Test sequences
        labels: Test labels
        device: Device to evaluate on

    Returns:
        Tuple of (predictions, probabilities)
    """
    logger.info("Evaluating CNN-GRU model...")

    model.eval()
    all_preds = []
    all_probs = []

    # Process in batches
    batch_size = 32
    num_samples = len(sequences)

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_seq = sequences[i:i + batch_size]
            batch_seq = torch.FloatTensor(batch_seq).to(device)

            outputs = model(batch_seq)
            probs = torch.softmax(outputs, dim=1)

            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_probs)


def evaluate_baseline(model, sequences: np.ndarray, labels: np.ndarray) -> tuple:
    """
    Evaluate baseline model.

    Args:
        model: Trained baseline model
        sequences: Test sequences
        labels: Test labels

    Returns:
        Tuple of (predictions, probabilities)
    """
    logger.info("Evaluating baseline model...")

    # Aggregate features
    X_test = aggregate_sequence_features(sequences)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    return y_pred, y_prob


def compare_models(results: dict, class_names: list = None):
    """
    Create comparison table of all models.

    Args:
        results: Dictionary of model results
        class_names: List of class names
    """
    logger.info("\n" + "="*80)
    logger.info("MODEL COMPARISON")
    logger.info("="*80)

    # Overall metrics comparison
    comparison_data = []

    for model_name, metrics in results.items():
        row = [
            model_name,
            f"{metrics['accuracy']:.4f}",
            f"{metrics['precision_macro']:.4f}",
            f"{metrics['recall_macro']:.4f}",
            f"{metrics['f1_macro']:.4f}",
            f"{metrics['f1_weighted']:.4f}"
        ]
        comparison_data.append(row)

    headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 (Macro)', 'F1 (Weighted)']
    table = tabulate(comparison_data, headers=headers, tablefmt='grid')

    print("\n" + table)
    logger.info("\n" + table)

    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    logger.info(f"\nBest Model: {best_model[0]} with accuracy {best_model[1]['accuracy']:.4f}")

    # Calculate improvement
    if 'CNN-GRU' in results and len(results) > 1:
        cnn_gru_acc = results['CNN-GRU']['accuracy']
        baseline_accs = [v['accuracy'] for k, v in results.items() if k != 'CNN-GRU']
        if baseline_accs:
            avg_baseline_acc = np.mean(baseline_accs)
            improvement = ((cnn_gru_acc - avg_baseline_acc) / avg_baseline_acc) * 100
            logger.info(f"\nCNN-GRU improvement over baselines: {improvement:.2f}%")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate and compare all models')
    parser.add_argument('--data_prefix', type=str, default='train',
                       help='Prefix for preprocessed data')
    parser.add_argument('--models', nargs='+',
                       default=['cnn_gru', 'svm', 'rf'],
                       help='Models to evaluate')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA')

    args = parser.parse_args()

    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading preprocessed data...")
    sequences, labels, student_ids = load_processed_data(args.data_prefix)

    # Split data
    train_data, val_data, test_data = split_data(
        sequences, labels, student_ids,
        val_split=TRAINING_CONFIG['validation_split'],
        test_split=TRAINING_CONFIG['test_split'],
        random_seed=TRAINING_CONFIG['random_seed']
    )

    X_test, y_test, _ = test_data
    logger.info(f"Test samples: {len(y_test)}")

    # Load class names
    try:
        import pickle
        preprocessor_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'processed', 'preprocessor.pkl'
        )
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        class_names = list(preprocessor['grade_encoder'].classes_)
    except:
        class_names = ['A', 'B', 'C', 'D', 'F']

    logger.info(f"Class names: {class_names}")

    # Evaluate models
    results = {}

    # CNN-GRU
    if 'cnn_gru' in args.models:
        cnn_gru_path = os.path.join(CHECKPOINTS_DIR, 'best_cnn_gru_model.pt')
        if os.path.exists(cnn_gru_path):
            try:
                model = load_cnn_gru_model(cnn_gru_path, device)
                y_pred, y_prob = evaluate_cnn_gru(model, X_test, y_test, device)
                metrics = calculate_metrics(y_test, y_pred, y_prob)
                results['CNN-GRU'] = metrics

                # Print detailed report
                logger.info("\nCNN-GRU Classification Report:")
                report = get_classification_report(y_test, y_pred, class_names)
                logger.info("\n" + report)

                # Confusion matrix
                cm = get_confusion_matrix(y_test, y_pred)
                logger.info("\nCNN-GRU Confusion Matrix:")
                logger.info(f"\n{cm}")

            except Exception as e:
                logger.error(f"Error evaluating CNN-GRU: {e}")
        else:
            logger.warning(f"CNN-GRU checkpoint not found at {cnn_gru_path}")

    # SVM
    if 'svm' in args.models:
        svm_path = os.path.join(CHECKPOINTS_DIR, 'svm_baseline.pkl')
        if os.path.exists(svm_path):
            try:
                model = load_baseline_model(svm_path)
                y_pred, y_prob = evaluate_baseline(model, X_test, y_test)
                metrics = calculate_metrics(y_test, y_pred, y_prob)
                results['SVM'] = metrics

                logger.info("\nSVM Classification Report:")
                report = get_classification_report(y_test, y_pred, class_names)
                logger.info("\n" + report)

            except Exception as e:
                logger.error(f"Error evaluating SVM: {e}")
        else:
            logger.warning(f"SVM model not found at {svm_path}")

    # Random Forest
    if 'rf' in args.models:
        rf_path = os.path.join(CHECKPOINTS_DIR, 'random_forest_baseline.pkl')
        if os.path.exists(rf_path):
            try:
                model = load_baseline_model(rf_path)
                y_pred, y_prob = evaluate_baseline(model, X_test, y_test)
                metrics = calculate_metrics(y_test, y_pred, y_prob)
                results['Random Forest'] = metrics

                logger.info("\nRandom Forest Classification Report:")
                report = get_classification_report(y_test, y_pred, class_names)
                logger.info("\n" + report)

            except Exception as e:
                logger.error(f"Error evaluating Random Forest: {e}")
        else:
            logger.warning(f"Random Forest model not found at {rf_path}")

    # Compare models
    if results:
        compare_models(results, class_names)

        # Save comparison
        comparison_path = os.path.join(METRICS_DIR, 'model_comparison.json')
        with open(comparison_path, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"\nComparison saved to {comparison_path}")
    else:
        logger.warning("No models were evaluated. Please train models first.")

    logger.info("\nEvaluation complete!")


if __name__ == '__main__':
    main()
