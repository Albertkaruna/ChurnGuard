import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 1. Setup MLflow
# Note: In K8s, we access MLflow via the service name or internal host
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("churn-prediction")

print(f"Tracking URI: {mlflow_uri}")

# 2. Generate Dummy Data (Simulating Telecom Churn)
# In real life, you'd read from MinIO: pd.read_csv('s3://data/churn.csv')
print("Generating data...")
np.random.seed(42)
n_samples = 1000
data = pd.DataFrame({
    'usage_minutes': np.random.normal(300, 100, n_samples),
    'customer_service_calls': np.random.poisson(2, n_samples),
    'monthly_charge': np.random.normal(50, 15, n_samples),
    'contract_type': np.random.choice([0, 1, 2], n_samples), # 0: Month-to-month, 1: 1yr, 2: 2yr
    'churn': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]) # 20% churn rate
})

X = data.drop('churn', axis=1)
y = data['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train & Log
print("Starting training...")
with mlflow.start_run():
    # Hyperparameters
    n_estimators = 100
    max_depth = 5

    # Log Params
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Train
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    print(f"Metrics: Acc={acc:.4f}, Prec={prec:.4f}, Recall={recall:.4f}")

    # Log Metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", recall)

    # Log Model
    mlflow.sklearn.log_model(clf, "model")
    print("Model saved to MLflow.")
