# 🎓 Academic Performance Intelligence System

A deep learning system that predicts **student academic risk and future performance** using sequential behavioral data from a real university course.

🔗 **Live Demo (Frontend Preview):**  
https://aruthrasathishkumar.github.io/academic-performance-predictor-CNN-GRU/frontend/index.html

> **Note:** This live demo is frontend-only (hosted on GitHub Pages).
- The backend (Python + PyTorch models + data pipeline) runs locally  
- Predictions are precomputed and loaded via `predictions.json`  
- Model training and updates require local execution  

💡 This design ensures **lightweight deployment + reproducibility without cloud dependency**.

## 🧠 What This Project Does

Traditional systems identify struggling students **too late**.

This system enables:

👉 **Early detection of at-risk students using behavioral patterns**

Instead of just grades, it analyzes:
- Grade trends over time  
- Assignment skip patterns  
- Time investment behavior  
- Learning velocity and volatility  

Example:

    Student: 87% average but inconsistent performance (100 → 0 → 100 → 0)

    System: Flags as "PATTERN RISK" due to volatility, not average

## ⚡ Key Features

- 📊 Predicts **at-risk probability** (classification)
- 📈 Predicts **next assignment grade** (regression)
- 🧠 Multi-task deep learning (shared CNN-GRU backbone)
- 🎯 Attention mechanism for **model explainability**
- ⚠️ PATTERN RISK detection (captures volatility-based risk)
- 🧾 Personalized intervention recommendations
- 📚 Assignment difficulty scoring from cohort behavior
- 📊 Interactive dashboard with real student insights

## 🏗️ Architecture

```
Codio LMS Data (CSV Export)
        │
        ▼
Data Processing (Pandas, NumPy)
        │
        ▼
Feature Engineering (13 behavioral features)
        │
        ▼
Deep Learning Pipeline
    ├── GRU v1 (multi-task + attention)
    └── CNN-GRU v2 (Conv1D + GRU + attention)
        │
        ▼
Predictions Export (predictions.json)
        │
        ▼
Frontend Dashboard (HTML + Plotly)
        │
        ▼
GitHub Pages (Frontend-only deployment)
```

## 🛠️ Tech Stack

| Layer | Technology | Why this choice |
|---|---|---|
| Data Processing | Pandas, NumPy | Efficient tabular transformations |
| ML Models | Scikit-learn | Baseline benchmarking |
| Deep Learning | PyTorch | Flexible sequence modeling |
| Tracking | MLflow | Experiment reproducibility |
| Visualization | Plotly | Interactive analytics dashboard |
| Frontend | HTML, CSS, JavaScript | Lightweight deployment |
| Deployment | GitHub Pages | Simple frontend hosting |

## ⚙️ How It Works

1. LMS data exported (grades + activity logs)  
2. Sequential features engineered per student  
3. Models trained on assignment sequences  
4. CNN extracts local learning patterns  
5. GRU captures temporal dependencies  
6. Attention highlights important assignments  
7. Predictions exported to JSON  
8. Dashboard visualizes insights  

## 📊 Model Results

| Model | Classification Accuracy | Regression MAE | R² |
|---|---|---|---|
| Logistic Regression | 86.5% | - | - |
| Random Forest | 84.1% | 18.07 pts | 0.391 |
| GRU v1 | **88.9%** | 22.18 pts | 0.197 |
| CNN-GRU v2 | 77.8% | **7.40 pts** | **0.853** |

### Key Insights

- ✅ GRU achieved highest classification accuracy (88.9%)  
- ✅ CNN-GRU improved regression error by **59%**  
- ⚠️ Risk probabilities are bimodal → reflects real engagement patterns  
- 🎯 Volatility-based risk detection outperforms average-based methods  

## 💡 System Design Decisions

### Why CNN-GRU?

- CNN captures **local patterns** (e.g., consecutive failures)  
- GRU captures **long-term trends**  
- Combined → better sequence understanding  

### Why Multi-task Learning?

- Shared representation improves both:
  - Grade prediction  
  - Risk classification  

### Why Attention?

- Provides **interpretability**  
- Shows *which assignments influenced predictions*  

### Why Frontend-only Deployment?

- Simplifies hosting  
- No backend dependency  
- Demonstrates results clearly  

## 📂 Project Structure

```
academic-performance-predictor-CNN-GRU/
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── feature_engineering.ipynb
│   ├── baseline_models.ipynb
│   ├── gru_model.ipynb
│   └── cnn_gru_model.ipynb
│
├── data/
│   └── processed/
│
├── outputs/
│   ├── models/
│   └── predictions/
│       └── predictions.json
│
├── frontend/
│   └── index.html
│
└── README.md
```

## 💻 Local Setup

### 1. Clone Repo

```bash
git clone https://github.com/Aruthrasathishkumar/academic-performance-predictor-CNN-GRU.git
cd academic-performance-predictor-CNN-GRU
```

### 2. Setup Environment

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run Dashboard

```bash
python -m http.server 8080
```

Open:

```
http://localhost:8080/frontend/
```

## 📊 Why This Project Stands Out

- Real-world dataset (actual university LMS data)  
- Advanced deep learning (CNN + GRU + Attention)  
- Multi-task learning architecture  
- Explainable AI (attention visualization)  
- Strong ML engineering (data cleaning + iteration)  
- Full pipeline: data → model → dashboard  

## 🚀 Future Improvements

- Real-time LMS integration (Codio API)
- Model calibration (temperature scaling)
- Larger multi-course dataset
- Student clustering (learning archetypes)
- Backend API for live predictions
- Instructor dashboard with alerts

## 🧩 Key Concept

> Student risk is not just about low grades - it’s about behavioral patterns over time.

## 📌 Summary

From:

❌ Static grade thresholds  
❌ Late intervention  
❌ Limited visibility  

To:

✅ Early risk detection  
✅ Behavioral pattern analysis  
✅ Actionable instructor insights  

This is not just an ML model.

This is a **real-world educational intelligence system designed to improve student outcomes at scale.**
