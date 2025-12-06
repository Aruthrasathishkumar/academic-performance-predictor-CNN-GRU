"""
Early Warning System - Run inference on new student data.
Predicts student performance risk levels and flags high-risk students.
"""

import torch
import numpy as np
import pandas as pd
import argparse
import os
from datetime import datetime

from models.cnn_gru_model import CNNGRU
from data_preprocessing import DataPreprocessor
from utils.config import (
    CNN_GRU_CONFIG, INFERENCE_CONFIG, PREDICTIONS_DIR, PROCESSED_DATA_DIR
)
from utils.logging_utils import get_default_logger

logger = get_default_logger('inference')


class EarlyWarningSystem:
    """Early Warning System for student performance prediction."""

    def __init__(self, model_path: str, preprocessor_path: str, device: torch.device):
        """
        Initialize the Early Warning System.

        Args:
            model_path: Path to trained model checkpoint
            preprocessor_path: Path to preprocessor pickle file
            device: Device to run inference on
        """
        self.device = device
        self.model = self._load_model(model_path)
        self.preprocessor = self._load_preprocessor(preprocessor_path)
        self.high_risk_threshold = INFERENCE_CONFIG['high_risk_threshold']

        # Risk mapping
        self.risk_mapping = {
            'A': 'Low Risk',
            'B': 'Low Risk',
            'C': 'Medium Risk',
            'D': 'High Risk',
            'F': 'High Risk'
        }

    def _load_model(self, model_path: str) -> CNNGRU:
        """Load trained CNN-GRU model."""
        logger.info(f"Loading model from {model_path}")

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

        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        logger.info("Model loaded successfully")
        return model

    def _load_preprocessor(self, preprocessor_path: str) -> DataPreprocessor:
        """Load preprocessor with fitted encoders and scalers."""
        logger.info(f"Loading preprocessor from {preprocessor_path}")

        preprocessor = DataPreprocessor()
        preprocessor.load_preprocessor(preprocessor_path)

        logger.info("Preprocessor loaded successfully")
        return preprocessor

    def preprocess_data(self, csv_path: str) -> tuple:
        """
        Preprocess new CSV data for inference.

        Args:
            csv_path: Path to CSV file with student activity logs

        Returns:
            Tuple of (sequences, student_ids, student_info)
        """
        logger.info(f"Preprocessing data from {csv_path}")

        # Load raw data
        df = self.preprocessor.load_raw_data(csv_path)
        df = self.preprocessor.clean_data(df)

        # Store student info for later
        student_info = df.groupby('student_id').first()[['grade']].to_dict('index')

        # Engineer features using existing encoders
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['time_since_last_action'] = df.groupby('student_id')['timestamp'].diff().dt.total_seconds()
        df['time_since_last_action'] = df['time_since_last_action'].fillna(0)

        # Transform using fitted encoders
        df['action_type_encoded'] = self.preprocessor.action_encoder.transform(df['action_type'])
        df['grade_encoded'] = self.preprocessor.grade_encoder.transform(df['grade'])

        # Create sequences
        sequences, labels, student_ids = self.preprocessor.create_sequences(df)

        # Normalize using fitted scaler
        sequences = self.preprocessor.normalize_sequences(sequences, fit=False)

        logger.info(f"Preprocessed {len(sequences)} student sequences")

        return sequences, student_ids, student_info

    def predict(self, sequences: np.ndarray) -> tuple:
        """
        Run inference on preprocessed sequences.

        Args:
            sequences: Preprocessed sequences

        Returns:
            Tuple of (predictions, probabilities)
        """
        logger.info(f"Running inference on {len(sequences)} sequences...")

        self.model.eval()
        all_preds = []
        all_probs = []

        batch_size = 32
        num_samples = len(sequences)

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch = sequences[i:i + batch_size]
                batch = torch.FloatTensor(batch).to(self.device)

                outputs = self.model(batch)
                probs = torch.softmax(outputs, dim=1)

                all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        predictions = np.array(all_preds)
        probabilities = np.array(all_probs)

        logger.info("Inference complete")

        return predictions, probabilities

    def generate_risk_report(self, student_ids: np.ndarray, predictions: np.ndarray,
                            probabilities: np.ndarray, student_info: dict) -> pd.DataFrame:
        """
        Generate risk assessment report for all students.

        Args:
            student_ids: Array of student IDs
            predictions: Predicted class indices
            probabilities: Prediction probabilities
            student_info: Dictionary with student information

        Returns:
            DataFrame with risk assessment
        """
        logger.info("Generating risk assessment report...")

        # Decode predictions
        grade_classes = self.preprocessor.grade_encoder.classes_
        predicted_grades = [grade_classes[p] for p in predictions]

        # Calculate risk scores and levels
        risk_scores = []
        risk_levels = []

        for i, (pred_grade, prob) in enumerate(zip(predicted_grades, probabilities)):
            # Risk score is the probability of being in D or F class
            d_idx = list(grade_classes).index('D') if 'D' in grade_classes else -1
            f_idx = list(grade_classes).index('F') if 'F' in grade_classes else -1

            risk_score = 0.0
            if d_idx >= 0:
                risk_score += prob[d_idx]
            if f_idx >= 0:
                risk_score += prob[f_idx]

            risk_scores.append(risk_score)

            # Determine risk level
            if risk_score >= self.high_risk_threshold:
                risk_level = 'High Risk'
            elif risk_score >= 0.3:
                risk_level = 'Medium Risk'
            else:
                risk_level = 'Low Risk'

            risk_levels.append(risk_level)

        # Create report DataFrame
        report = pd.DataFrame({
            'student_id': student_ids,
            'predicted_grade': predicted_grades,
            'risk_level': risk_levels,
            'risk_score': risk_scores,
            'confidence': [prob[pred] for pred, prob in zip(predictions, probabilities)]
        })

        # Add actual grades if available
        if student_info:
            report['actual_grade'] = report['student_id'].map(
                lambda x: student_info.get(x, {}).get('grade', 'N/A')
            )

        # Sort by risk score (descending)
        report = report.sort_values('risk_score', ascending=False)

        logger.info(f"Risk report generated for {len(report)} students")

        return report

    def flag_high_risk_students(self, report: pd.DataFrame) -> pd.DataFrame:
        """
        Flag high-risk students for intervention.

        Args:
            report: Risk assessment report

        Returns:
            DataFrame with only high-risk students
        """
        high_risk = report[report['risk_level'] == 'High Risk'].copy()

        logger.info(f"Flagged {len(high_risk)} high-risk students")

        return high_risk

    def save_predictions(self, report: pd.DataFrame, output_path: str):
        """
        Save prediction report to CSV.

        Args:
            report: Risk assessment report
            output_path: Path to save the report
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        report.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")

    def run_inference_pipeline(self, csv_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Run complete inference pipeline.

        Args:
            csv_path: Path to input CSV file
            output_path: Path to save predictions (optional)

        Returns:
            Risk assessment report
        """
        # Preprocess
        sequences, student_ids, student_info = self.preprocess_data(csv_path)

        # Predict
        predictions, probabilities = self.predict(sequences)

        # Generate report
        report = self.generate_risk_report(student_ids, predictions, probabilities, student_info)

        # Flag high-risk students
        high_risk = self.flag_high_risk_students(report)

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("EARLY WARNING SYSTEM SUMMARY")
        logger.info("="*60)
        logger.info(f"Total students analyzed: {len(report)}")
        logger.info(f"High-risk students: {len(high_risk)}")
        logger.info(f"Medium-risk students: {len(report[report['risk_level'] == 'Medium Risk'])}")
        logger.info(f"Low-risk students: {len(report[report['risk_level'] == 'Low Risk'])}")
        logger.info("="*60)

        # Print high-risk students
        if len(high_risk) > 0:
            logger.info("\nHIGH-RISK STUDENTS REQUIRING INTERVENTION:")
            logger.info("-"*60)
            for _, row in high_risk.iterrows():
                logger.info(f"Student {row['student_id']}: {row['predicted_grade']} "
                          f"(Risk: {row['risk_score']:.2f}, Confidence: {row['confidence']:.2f})")

        # Save
        if output_path:
            self.save_predictions(report, output_path)

        return report


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Early Warning System - Student Performance Prediction')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to CSV file with student activity logs')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save predictions (default: outputs/predictions/predictions_[timestamp].csv)')
    parser.add_argument('--model', type=str, default=INFERENCE_CONFIG['model_checkpoint'],
                       help='Path to trained model checkpoint')
    parser.add_argument('--preprocessor', type=str,
                       default=os.path.join(PROCESSED_DATA_DIR, 'preprocessor.pkl'),
                       help='Path to preprocessor pickle file')
    parser.add_argument('--threshold', type=float, default=INFERENCE_CONFIG['high_risk_threshold'],
                       help='High-risk threshold (0-1)')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA')

    args = parser.parse_args()

    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"Using device: {device}")

    # Set output path
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = os.path.join(PREDICTIONS_DIR, f'predictions_{timestamp}.csv')

    # Initialize Early Warning System
    ews = EarlyWarningSystem(args.model, args.preprocessor, device)
    ews.high_risk_threshold = args.threshold

    # Run inference
    logger.info(f"Starting inference on {args.input}")
    report = ews.run_inference_pipeline(args.input, args.output)

    logger.info("\nInference complete!")
    logger.info(f"Full report saved to {args.output}")


if __name__ == '__main__':
    main()
