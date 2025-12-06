"""
PyTorch Dataset classes for student activity sequences.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from typing import Tuple
from utils.config import PROCESSED_DATA_DIR, TRAINING_CONFIG


class StudentSequenceDataset(Dataset):
    """PyTorch Dataset for student activity sequences."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray,
                 student_ids: np.ndarray = None):
        """
        Initialize the dataset.

        Args:
            sequences: NumPy array of shape (num_students, seq_len, num_features)
            labels: NumPy array of shape (num_students,)
            student_ids: Optional student IDs
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.student_ids = student_ids if student_ids is not None else np.arange(len(sequences))

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (sequence, label)
        """
        return self.sequences[idx], self.labels[idx]

    def get_student_id(self, idx: int) -> int:
        """Get student ID for a given index."""
        return self.student_ids[idx]


def load_processed_data(prefix: str = 'train') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load preprocessed data from disk.

    Args:
        prefix: Data prefix (train/val/test)

    Returns:
        Tuple of (sequences, labels, student_ids)
    """
    sequences = np.load(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_sequences.npy'))
    labels = np.load(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_labels.npy'))
    student_ids = np.load(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_student_ids.npy'))

    return sequences, labels, student_ids


def create_data_loaders(train_sequences: np.ndarray, train_labels: np.ndarray,
                       val_sequences: np.ndarray = None, val_labels: np.ndarray = None,
                       batch_size: int = TRAINING_CONFIG['batch_size'],
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation.

    Args:
        train_sequences: Training sequences
        train_labels: Training labels
        val_sequences: Validation sequences (optional)
        val_labels: Validation labels (optional)
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = StudentSequenceDataset(train_sequences, train_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = None
    if val_sequences is not None and val_labels is not None:
        val_dataset = StudentSequenceDataset(val_sequences, val_labels)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

    return train_loader, val_loader


def split_data(sequences: np.ndarray, labels: np.ndarray, student_ids: np.ndarray,
               val_split: float = 0.2, test_split: float = 0.1,
               random_seed: int = 42) -> Tuple:
    """
    Split data into train, validation, and test sets.

    Args:
        sequences: All sequences
        labels: All labels
        student_ids: All student IDs
        val_split: Validation split ratio
        test_split: Test split ratio
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, val_data, test_data), where each is (sequences, labels, student_ids)
    """
    np.random.seed(random_seed)

    num_samples = len(sequences)
    indices = np.random.permutation(num_samples)

    # Calculate split sizes
    test_size = int(num_samples * test_split)
    val_size = int(num_samples * val_split)
    train_size = num_samples - test_size - val_size

    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Split data
    train_data = (
        sequences[train_indices],
        labels[train_indices],
        student_ids[train_indices]
    )

    val_data = (
        sequences[val_indices],
        labels[val_indices],
        student_ids[val_indices]
    )

    test_data = (
        sequences[test_indices],
        labels[test_indices],
        student_ids[test_indices]
    )

    return train_data, val_data, test_data


def get_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.

    Args:
        labels: Training labels

    Returns:
        Tensor of class weights
    """
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)

    # Calculate weights inversely proportional to class frequency
    weights = total_samples / (num_classes * class_counts)
    weights = torch.FloatTensor(weights)

    return weights
