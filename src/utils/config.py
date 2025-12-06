"""
Configuration file for the Academic Performance Predictor project.
Contains all hyperparameters and project settings.
"""

import os

# ==================== PATHS ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
CHECKPOINTS_DIR = os.path.join(OUTPUTS_DIR, 'checkpoints')
METRICS_DIR = os.path.join(OUTPUTS_DIR, 'metrics')
PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, 'predictions')

# ==================== DATA GENERATION ====================
SYNTHETIC_DATA_CONFIG = {
    'num_students': 500,
    'num_assignments': 10,
    'min_actions_per_student': 50,
    'max_actions_per_student': 300,
    'action_types': ['run', 'compile', 'test', 'save', 'debug'],
    'grade_bands': ['A', 'B', 'C', 'D', 'F'],  # 5 classes
    'risk_mapping': {
        'A': 'low',
        'B': 'low',
        'C': 'medium',
        'D': 'high',
        'F': 'high'
    }
}

# ==================== PREPROCESSING ====================
SEQUENCE_LENGTH = 100  # Fixed sequence length for padding/truncating
FEATURE_COLUMNS = [
    'action_type_encoded',
    'error_count',
    'test_passed',
    'assignment_id',
    'hour_of_day',
    'day_of_week',
    'time_since_last_action'
]

# ==================== MODEL HYPERPARAMETERS ====================
CNN_GRU_CONFIG = {
    'input_dim': len(FEATURE_COLUMNS),
    'cnn_channels': [64, 128, 128],  # Multi-layer CNN
    'kernel_sizes': [3, 3, 3],
    'gru_hidden_dim': 256,
    'gru_num_layers': 2,
    'dropout': 0.3,
    'num_classes': len(SYNTHETIC_DATA_CONFIG['grade_bands']),
    'bidirectional': True
}

# ==================== TRAINING ====================
TRAINING_CONFIG = {
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'validation_split': 0.2,
    'test_split': 0.1,
    'early_stopping_patience': 10,
    'random_seed': 42
}

# ==================== BASELINE MODELS ====================
BASELINE_CONFIG = {
    'svm': {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale'
    },
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 5,
        'random_state': TRAINING_CONFIG['random_seed']
    }
}

# ==================== INFERENCE ====================
INFERENCE_CONFIG = {
    'high_risk_threshold': 0.6,  # Probability threshold for high-risk classification
    'model_checkpoint': os.path.join(CHECKPOINTS_DIR, 'best_cnn_gru_model.pt')
}

# ==================== LOGGING ====================
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'
