"""
Data preprocessing module for student activity logs.
Handles cleaning, feature engineering, and sequence building.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os
import argparse
from typing import Tuple, Dict
from utils.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, SEQUENCE_LENGTH,
    FEATURE_COLUMNS, SYNTHETIC_DATA_CONFIG
)
from utils.logging_utils import setup_logger

logger = setup_logger('data_preprocessing')


class DataPreprocessor:
    """Preprocess student activity logs into sequences for deep learning."""

    def __init__(self, sequence_length: int = SEQUENCE_LENGTH):
        """
        Initialize the preprocessor.

        Args:
            sequence_length: Fixed length for all sequences
        """
        self.sequence_length = sequence_length
        self.action_encoder = LabelEncoder()
        self.grade_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = []

    def load_raw_data(self, filepath: str) -> pd.DataFrame:
        """
        Load raw CSV data.

        Args:
            filepath: Path to CSV file

        Returns:
            DataFrame with raw data
        """
        logger.info(f"Loading raw data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records for {df['student_id'].nunique()} students")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate data.

        Args:
            df: Raw dataframe

        Returns:
            Cleaned dataframe
        """
        logger.info("Cleaning data...")

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Handle missing values
        df = df.dropna(subset=['student_id', 'timestamp', 'action_type'])

        # Fill missing numeric values
        df['error_count'] = df['error_count'].fillna(0)
        df['test_passed'] = df['test_passed'].fillna(-1)  # -1 for N/A

        # Sort by student and time
        df = df.sort_values(['student_id', 'timestamp']).reset_index(drop=True)

        logger.info(f"Data cleaned. {len(df)} records remaining")
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features from raw data.

        Args:
            df: Cleaned dataframe

        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features...")

        # Time-based features
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek

        # Calculate time since last action (per student)
        df['time_since_last_action'] = df.groupby('student_id')['timestamp'].diff().dt.total_seconds()
        df['time_since_last_action'] = df['time_since_last_action'].fillna(0)

        # Encode action types
        df['action_type_encoded'] = self.action_encoder.fit_transform(df['action_type'])

        # Encode grades (labels)
        df['grade_encoded'] = self.grade_encoder.fit_transform(df['grade'])

        logger.info(f"Features engineered. Action types: {list(self.action_encoder.classes_)}")
        logger.info(f"Grade classes: {list(self.grade_encoder.classes_)}")

        return df

    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create fixed-length sequences for each student.

        Args:
            df: DataFrame with features

        Returns:
            Tuple of (sequences, labels, student_ids)
        """
        logger.info(f"Creating sequences of length {self.sequence_length}...")

        feature_cols = [
            'action_type_encoded', 'error_count', 'test_passed',
            'assignment_id', 'hour_of_day', 'day_of_week',
            'time_since_last_action'
        ]

        self.feature_names = feature_cols

        sequences = []
        labels = []
        student_ids = []

        for student_id in df['student_id'].unique():
            student_data = df[df['student_id'] == student_id]

            # Extract features
            features = student_data[feature_cols].values

            # Pad or truncate to fixed length
            if len(features) < self.sequence_length:
                # Pad with zeros at the beginning
                padding = np.zeros((self.sequence_length - len(features), len(feature_cols)))
                features = np.vstack([padding, features])
            else:
                # Take the most recent sequence_length actions
                features = features[-self.sequence_length:]

            sequences.append(features)
            labels.append(student_data['grade_encoded'].iloc[0])  # All rows have same grade
            student_ids.append(student_id)

        sequences = np.array(sequences, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        student_ids = np.array(student_ids, dtype=np.int64)

        logger.info(f"Created {len(sequences)} sequences")
        logger.info(f"Sequence shape: {sequences.shape}")
        logger.info(f"Label distribution: {np.bincount(labels)}")

        return sequences, labels, student_ids

    def normalize_sequences(self, sequences: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize feature sequences.

        Args:
            sequences: Input sequences (num_students, seq_len, num_features)
            fit: Whether to fit the scaler (True for training, False for test/inference)

        Returns:
            Normalized sequences
        """
        logger.info("Normalizing sequences...")

        original_shape = sequences.shape
        # Reshape to (num_students * seq_len, num_features)
        sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])

        if fit:
            sequences_normalized = self.scaler.fit_transform(sequences_reshaped)
        else:
            sequences_normalized = self.scaler.transform(sequences_reshaped)

        # Reshape back
        sequences_normalized = sequences_normalized.reshape(original_shape)

        return sequences_normalized.astype(np.float32)

    def save_preprocessed_data(self, sequences: np.ndarray, labels: np.ndarray,
                               student_ids: np.ndarray, prefix: str = 'train'):
        """
        Save preprocessed data and preprocessing objects.

        Args:
            sequences: Feature sequences
            labels: Target labels
            student_ids: Student identifiers
            prefix: Prefix for output files (train/test/val)
        """
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

        # Save data
        np.save(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_sequences.npy'), sequences)
        np.save(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_labels.npy'), labels)
        np.save(os.path.join(PROCESSED_DATA_DIR, f'{prefix}_student_ids.npy'), student_ids)

        # Save preprocessing objects (only for training data)
        if prefix == 'train':
            with open(os.path.join(PROCESSED_DATA_DIR, 'preprocessor.pkl'), 'wb') as f:
                pickle.dump({
                    'action_encoder': self.action_encoder,
                    'grade_encoder': self.grade_encoder,
                    'scaler': self.scaler,
                    'feature_names': self.feature_names,
                    'sequence_length': self.sequence_length
                }, f)

        logger.info(f"Saved preprocessed data with prefix '{prefix}'")

    def load_preprocessor(self, filepath: str = None):
        """
        Load saved preprocessing objects.

        Args:
            filepath: Path to preprocessor pickle file
        """
        if filepath is None:
            filepath = os.path.join(PROCESSED_DATA_DIR, 'preprocessor.pkl')

        with open(filepath, 'rb') as f:
            obj = pickle.load(f)

        self.action_encoder = obj['action_encoder']
        self.grade_encoder = obj['grade_encoder']
        self.scaler = obj['scaler']
        self.feature_names = obj['feature_names']
        self.sequence_length = obj['sequence_length']

        logger.info(f"Loaded preprocessor from {filepath}")

    def process_pipeline(self, input_file: str, output_prefix: str = 'train',
                        fit_preprocessor: bool = True) -> Dict:
        """
        Run the complete preprocessing pipeline.

        Args:
            input_file: Path to raw CSV file
            output_prefix: Prefix for output files
            fit_preprocessor: Whether to fit preprocessing objects

        Returns:
            Dictionary with processing statistics
        """
        # Load and clean
        df = self.load_raw_data(input_file)
        df = self.clean_data(df)

        # Engineer features
        if fit_preprocessor:
            df = self.engineer_features(df)
        else:
            # Use existing encoders
            df['hour_of_day'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['time_since_last_action'] = df.groupby('student_id')['timestamp'].diff().dt.total_seconds()
            df['time_since_last_action'] = df['time_since_last_action'].fillna(0)
            df['action_type_encoded'] = self.action_encoder.transform(df['action_type'])
            df['grade_encoded'] = self.grade_encoder.transform(df['grade'])

        # Create sequences
        sequences, labels, student_ids = self.create_sequences(df)

        # Normalize
        sequences = self.normalize_sequences(sequences, fit=fit_preprocessor)

        # Save
        self.save_preprocessed_data(sequences, labels, student_ids, output_prefix)

        stats = {
            'num_students': len(student_ids),
            'num_features': sequences.shape[-1],
            'sequence_length': sequences.shape[1],
            'num_classes': len(np.unique(labels)),
            'class_distribution': {
                self.grade_encoder.classes_[i]: int(np.sum(labels == i))
                for i in range(len(self.grade_encoder.classes_))
            }
        }

        return stats


def main():
    """Main function for data preprocessing."""
    parser = argparse.ArgumentParser(description='Preprocess student activity logs')
    parser.add_argument('--input', type=str,
                       default=os.path.join(RAW_DATA_DIR, 'student_activity_logs.csv'),
                       help='Input CSV file path')
    parser.add_argument('--output_prefix', type=str, default='train',
                       help='Prefix for output files')
    parser.add_argument('--sequence_length', type=int, default=SEQUENCE_LENGTH,
                       help='Fixed sequence length')

    args = parser.parse_args()

    # Run preprocessing
    preprocessor = DataPreprocessor(sequence_length=args.sequence_length)
    stats = preprocessor.process_pipeline(args.input, args.output_prefix)

    logger.info("\n" + "="*60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("="*60)
    for key, value in stats.items():
        logger.info(f"{key}: {value}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
