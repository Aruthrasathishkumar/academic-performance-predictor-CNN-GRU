import pandas as pd
import numpy as np
import sklearn
import torch
import plotly
import mlflow

print("Setup verification")
print("==================")
print(f"Pandas    : {pd.__version__}")
print(f"NumPy     : {np.__version__}")
print(f"Sklearn   : {sklearn.__version__}")
print(f"PyTorch   : {torch.__version__}")
print(f"Plotly    : {plotly.__version__}")
print(f"MLflow    : {mlflow.__version__}")
print()
print("All libraries imported successfully.")
print("Phase 0 complete.")