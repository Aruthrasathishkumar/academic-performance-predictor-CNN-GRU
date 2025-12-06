# Quick Start Guide

Get the Academic Performance Predictor up and running in **5 minutes**.

## Prerequisites

- Python 3.8+ installed
- Terminal/Command Prompt access

## Step-by-Step Instructions

### 1. Setup Virtual Environment (REQUIRED)

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Complete Pipeline

Execute these commands in order:

```bash
# Step 1: Generate synthetic data (500 students)
python src/synthetic_data_generator.py

# Step 2: Preprocess data into sequences
python src/data_preprocessing.py

# Step 3: Train baseline models (takes ~1 minute)
python src/train_baselines.py --models both --save_models

# Step 4: Train CNN-GRU model (takes ~5-10 minutes)
python src/train_cnn_gru.py --epochs 50

# Step 5: Compare all models
python src/evaluate_models.py

# Step 6: Run inference (Early Warning System)
python src/run_inference.py --input data/raw/student_activity_logs.csv
```

## Expected Timeline

| Step | Time | Output |
|------|------|--------|
| Data generation | 10s | 500 students, ~100k actions |
| Preprocessing | 5s | Normalized sequences |
| Baseline training | 1 min | SVM + Random Forest models |
| CNN-GRU training | 5-10 min | Deep learning model |
| Evaluation | 10s | Comparison metrics |
| Inference | 5s | Risk predictions |

**Total:** ~15 minutes for full pipeline

## Verify Success

After running all steps, you should have:

```
outputs/
├── checkpoints/
│   ├── best_cnn_gru_model.pt       ✓
│   ├── svm_baseline.pkl            ✓
│   └── random_forest_baseline.pkl  ✓
├── metrics/
│   ├── cnn_gru_test_metrics.json   ✓
│   ├── svm_metrics.json            ✓
│   ├── random_forest_metrics.json  ✓
│   └── model_comparison.json       ✓
└── predictions/
    └── predictions_*.csv           ✓
```

## Expected Results

**Model Performance:**
- SVM: ~75% accuracy
- Random Forest: ~78% accuracy
- CNN-GRU: **~89% accuracy** ⭐

**Early Warning System:**
- Identifies ~15-20% of students as high-risk
- Flags students with >60% probability of D/F grades

## Troubleshooting

### "ModuleNotFoundError"
- Make sure virtual environment is activated: `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows)
- Reinstall dependencies: `pip install -r requirements.txt`

### "FileNotFoundError"
- Run steps in order (data generation → preprocessing → training)
- Check that `data/raw/student_activity_logs.csv` exists

### "CUDA out of memory"
- Add `--no_cuda` flag to training commands
- Reduce batch size: `--batch_size 16`

### Slow training
- Reduce epochs: `--epochs 30`
- Use smaller dataset: `--num_students 200` when generating data

## Next Steps

1. **Explore the data:** Check `data/raw/dataset_summary.txt`
2. **Analyze predictions:** Open `outputs/predictions/predictions_*.csv`
3. **Customize model:** Edit `src/utils/config.py` to change hyperparameters
4. **Use your own data:** Replace synthetic data with real Codio/LMS logs

## Questions?

See the full [README.md](README.md) for detailed documentation.
