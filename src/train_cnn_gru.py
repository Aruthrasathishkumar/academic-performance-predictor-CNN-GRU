"""
Train the CNN-GRU model for academic performance prediction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import time
from tqdm import tqdm

from models.cnn_gru_model import CNNGRU, count_parameters
from dataset import (
    load_processed_data, split_data, create_data_loaders, get_class_weights
)
from utils.config import (
    CNN_GRU_CONFIG, TRAINING_CONFIG, CHECKPOINTS_DIR, METRICS_DIR
)
from utils.metrics import calculate_metrics, print_metrics, save_metrics
from utils.logging_utils import get_default_logger, TrainingLogger

logger = get_default_logger('train_cnn_gru')


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(self, patience=10, min_delta=0.0):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Average loss and accuracy
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for sequences, labels in train_loader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item() * sequences.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def validate_epoch(model, val_loader, criterion, device):
    """
    Validate for one epoch.

    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Average loss and accuracy
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            # Statistics
            total_loss += loss.item() * sequences.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")


def train_model(model, train_loader, val_loader, num_epochs, device,
                criterion, optimizer, scheduler=None, early_stopping=None):
    """
    Complete training loop.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs
        device: Device to train on
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        early_stopping: Early stopping object (optional)

    Returns:
        Training history
    """
    training_logger = TrainingLogger(logger)
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()

        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validation phase
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        # Update learning rate
        if scheduler is not None:
            scheduler.step(val_loss)

        # Record metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Log epoch
        training_logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc)

        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch time: {epoch_time:.2f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, {'val_acc': val_acc, 'val_loss': val_loss},
                os.path.join(CHECKPOINTS_DIR, 'best_cnn_gru_model.pt')
            )
            training_logger.log_best_model(epoch, val_acc, 'validation accuracy')

        # Early stopping
        if early_stopping is not None:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                training_logger.log_early_stopping(epoch, early_stopping.patience)
                break

    return history


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train CNN-GRU model')
    parser.add_argument('--data_prefix', type=str, default='train',
                       help='Prefix for preprocessed data')
    parser.add_argument('--epochs', type=int, default=TRAINING_CONFIG['num_epochs'],
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=TRAINING_CONFIG['batch_size'],
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=TRAINING_CONFIG['learning_rate'],
                       help='Learning rate')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA training')
    parser.add_argument('--early_stopping', type=int,
                       default=TRAINING_CONFIG['early_stopping_patience'],
                       help='Early stopping patience (0 to disable)')

    args = parser.parse_args()

    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"Using device: {device}")

    # Set random seed
    set_seed(TRAINING_CONFIG['random_seed'])

    # Create output directories
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # Load data
    logger.info("Loading preprocessed data...")
    sequences, labels, student_ids = load_processed_data(args.data_prefix)

    logger.info(f"Data shape: {sequences.shape}")
    logger.info(f"Number of classes: {len(np.unique(labels))}")

    # Split data
    logger.info("Splitting data into train/val/test sets...")
    train_data, val_data, test_data = split_data(
        sequences, labels, student_ids,
        val_split=TRAINING_CONFIG['validation_split'],
        test_split=TRAINING_CONFIG['test_split'],
        random_seed=TRAINING_CONFIG['random_seed']
    )

    X_train, y_train, _ = train_data
    X_val, y_val, _ = val_data
    X_test, y_test, _ = test_data

    logger.info(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_val, y_val,
        batch_size=args.batch_size
    )

    # Initialize model
    logger.info("Initializing CNN-GRU model...")
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

    model = model.to(device)
    num_params = count_parameters(model)
    logger.info(f"Model has {num_params:,} trainable parameters")

    # Loss function with class weights for imbalanced data
    class_weights = get_class_weights(y_train).to(device)
    logger.info(f"Class weights: {class_weights}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=TRAINING_CONFIG['weight_decay']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Early stopping
    early_stopping = None
    if args.early_stopping > 0:
        early_stopping = EarlyStopping(patience=args.early_stopping)

    # Train model
    logger.info(f"Starting training for {args.epochs} epochs...")
    logger.info("="*60)

    history = train_model(
        model, train_loader, val_loader, args.epochs, device,
        criterion, optimizer, scheduler, early_stopping
    )

    logger.info("="*60)
    logger.info("Training completed!")

    # Save training history
    history_path = os.path.join(METRICS_DIR, 'cnn_gru_training_history.npy')
    np.save(history_path, history)
    logger.info(f"Training history saved to {history_path}")

    # Load best model for final evaluation
    logger.info("Loading best model for evaluation...")
    checkpoint = torch.load(os.path.join(CHECKPOINTS_DIR, 'best_cnn_gru_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_loader, _ = create_data_loaders(X_test, y_test, batch_size=args.batch_size)

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            probs = torch.softmax(outputs, dim=1)

            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    test_metrics = calculate_metrics(all_labels, all_preds, all_probs)
    print_metrics(test_metrics, "CNN-GRU Test Set")

    # Save test metrics
    save_metrics(test_metrics, os.path.join(METRICS_DIR, 'cnn_gru_test_metrics.json'))

    logger.info("\nAll training and evaluation complete!")


if __name__ == '__main__':
    main()
