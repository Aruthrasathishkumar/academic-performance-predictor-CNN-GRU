"""
Synthetic data generator for Codio-style coding activity logs.
Generates realistic student coding behavior patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import argparse
from utils.config import SYNTHETIC_DATA_CONFIG, RAW_DATA_DIR
from utils.logging_utils import setup_logger

logger = setup_logger('synthetic_data_generator')


class SyntheticDataGenerator:
    """Generate synthetic Codio-style student activity logs."""

    def __init__(self, config=SYNTHETIC_DATA_CONFIG, seed=42):
        """
        Initialize the synthetic data generator.

        Args:
            config: Configuration dictionary
            seed: Random seed for reproducibility
        """
        self.config = config
        self.seed = seed
        np.random.seed(seed)

        self.num_students = config['num_students']
        self.num_assignments = config['num_assignments']
        self.action_types = config['action_types']
        self.grade_bands = config['grade_bands']

    def generate_student_profile(self, student_id: int) -> dict:
        """
        Generate a student profile that determines their behavior pattern.

        Args:
            student_id: Unique student identifier

        Returns:
            Dictionary with student characteristics
        """
        # Assign a performance level (determines final grade)
        performance_level = np.random.choice([0, 1, 2, 3, 4], p=[0.15, 0.25, 0.30, 0.20, 0.10])
        grade = self.grade_bands[performance_level]

        # Student characteristics based on performance
        profiles = {
            0: {  # A students
                'grade': 'A',
                'avg_error_rate': 0.05,
                'test_pass_rate': 0.95,
                'activity_frequency': 'high',
                'consistency': 0.9
            },
            1: {  # B students
                'grade': 'B',
                'avg_error_rate': 0.15,
                'test_pass_rate': 0.85,
                'activity_frequency': 'medium-high',
                'consistency': 0.75
            },
            2: {  # C students
                'grade': 'C',
                'avg_error_rate': 0.30,
                'test_pass_rate': 0.70,
                'activity_frequency': 'medium',
                'consistency': 0.60
            },
            3: {  # D students
                'grade': 'D',
                'avg_error_rate': 0.50,
                'test_pass_rate': 0.50,
                'activity_frequency': 'low',
                'consistency': 0.40
            },
            4: {  # F students
                'grade': 'F',
                'avg_error_rate': 0.70,
                'test_pass_rate': 0.30,
                'activity_frequency': 'very-low',
                'consistency': 0.20
            }
        }

        return profiles[performance_level]

    def generate_student_actions(self, student_id: int, profile: dict) -> pd.DataFrame:
        """
        Generate action logs for a single student.

        Args:
            student_id: Student ID
            profile: Student profile dictionary

        Returns:
            DataFrame with student actions
        """
        # Determine number of actions based on profile
        activity_ranges = {
            'very-low': (50, 100),
            'low': (100, 150),
            'medium': (150, 200),
            'medium-high': (200, 250),
            'high': (250, 300)
        }

        min_actions, max_actions = activity_ranges[profile['activity_frequency']]
        num_actions = np.random.randint(min_actions, max_actions)

        actions = []
        base_time = datetime.now() - timedelta(days=90)  # Start 90 days ago

        for i in range(num_actions):
            # Generate timestamp (with some consistency based on profile)
            if np.random.random() < profile['consistency']:
                # Consistent students work during normal hours
                hour = np.random.randint(9, 22)
                day_offset = i // 10  # Spread over time
            else:
                # Inconsistent students have erratic patterns
                hour = np.random.randint(0, 24)
                day_offset = np.random.randint(0, 90)

            timestamp = base_time + timedelta(days=day_offset, hours=hour,
                                             minutes=np.random.randint(0, 60))

            # Assign to an assignment
            assignment_id = np.random.randint(1, self.num_assignments + 1)

            # Generate action type (students tend to run/test more than save)
            action_type = np.random.choice(
                self.action_types,
                p=[0.30, 0.25, 0.25, 0.15, 0.05]  # run, compile, test, save, debug
            )

            # Generate error count based on profile
            if action_type in ['run', 'compile', 'test']:
                error_count = np.random.poisson(profile['avg_error_rate'] * 10)
            else:
                error_count = 0

            # Generate test pass/fail
            if action_type == 'test':
                test_passed = 1 if np.random.random() < profile['test_pass_rate'] else 0
            else:
                test_passed = -1  # Not applicable

            actions.append({
                'student_id': student_id,
                'timestamp': timestamp,
                'action_type': action_type,
                'error_count': error_count,
                'test_passed': test_passed,
                'assignment_id': assignment_id,
                'grade': profile['grade']
            })

        return pd.DataFrame(actions)

    def generate_dataset(self) -> pd.DataFrame:
        """
        Generate complete synthetic dataset for all students.

        Returns:
            DataFrame with all student actions
        """
        logger.info(f"Generating synthetic data for {self.num_students} students...")

        all_data = []

        for student_id in range(1, self.num_students + 1):
            if student_id % 100 == 0:
                logger.info(f"Generated data for {student_id}/{self.num_students} students")

            profile = self.generate_student_profile(student_id)
            student_data = self.generate_student_actions(student_id, profile)
            all_data.append(student_data)

        # Combine all student data
        dataset = pd.concat(all_data, ignore_index=True)

        # Sort by student and timestamp
        dataset = dataset.sort_values(['student_id', 'timestamp']).reset_index(drop=True)

        logger.info(f"Generated {len(dataset)} total actions for {self.num_students} students")
        logger.info(f"Grade distribution:\n{dataset.groupby('grade')['student_id'].nunique()}")

        return dataset

    def save_dataset(self, dataset: pd.DataFrame, filename: str = 'student_activity_logs.csv'):
        """
        Save dataset to CSV file.

        Args:
            dataset: DataFrame to save
            filename: Output filename
        """
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        filepath = os.path.join(RAW_DATA_DIR, filename)

        dataset.to_csv(filepath, index=False)
        logger.info(f"Dataset saved to {filepath}")

        # Save summary statistics
        summary_path = os.path.join(RAW_DATA_DIR, 'dataset_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("SYNTHETIC DATASET SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total students: {dataset['student_id'].nunique()}\n")
            f.write(f"Total actions: {len(dataset)}\n")
            f.write(f"Date range: {dataset['timestamp'].min()} to {dataset['timestamp'].max()}\n\n")
            f.write("Grade distribution:\n")
            f.write(str(dataset.groupby('grade')['student_id'].nunique()) + "\n\n")
            f.write("Action type distribution:\n")
            f.write(str(dataset['action_type'].value_counts()) + "\n\n")
            f.write(f"Average actions per student: {len(dataset) / dataset['student_id'].nunique():.2f}\n")

        logger.info(f"Summary saved to {summary_path}")


def main():
    """Main function to generate synthetic data."""
    parser = argparse.ArgumentParser(description='Generate synthetic Codio-style student activity logs')
    parser.add_argument('--num_students', type=int, default=SYNTHETIC_DATA_CONFIG['num_students'],
                       help='Number of students to generate')
    parser.add_argument('--output', type=str, default='student_activity_logs.csv',
                       help='Output filename')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Update config with command line arguments
    config = SYNTHETIC_DATA_CONFIG.copy()
    config['num_students'] = args.num_students

    # Generate data
    generator = SyntheticDataGenerator(config, seed=args.seed)
    dataset = generator.generate_dataset()
    generator.save_dataset(dataset, args.output)

    logger.info("Synthetic data generation complete!")


if __name__ == '__main__':
    main()
