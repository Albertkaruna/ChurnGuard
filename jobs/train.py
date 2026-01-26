"""
ML Model Training Script with MLflow Integration
Trains a model and logs to MLflow
"""
import os
from datetime import datetime
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

def train_model():
    """Train a simple ML model and log to MLflow"""
    print(f"Starting model training at {datetime.now()}")
    
    # Set MLflow tracking URI - using Kubernetes service DNS
    # Format: http://<service-name>.<namespace>.svc.cluster.local:<port>
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://10.96.84.133:80:80') # http://service_name.namespace_name.svc.cluster.local:PORT
    mlflow.set_tracking_uri(mlflow_uri)
    print(f"MLflow tracking URI: {mlflow_uri}")
    
    os.environ['MLFLOW_TRACKING_USERNAME'] = "user"
    os.environ['MLFLOW_TRACKING_PASSWORD'] = "S5z2iGZPKjmJ"

    # Set experiment name
    experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'churn-prediction')
    mlflow.set_experiment(experiment_name)
    print(f"MLflow experiment: {experiment_name}")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log parameters
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'n_samples': 1000,
            'n_features': 20,
        }
        mlflow.log_params(params)
        
        # Generate sample data
        print("Generating training data...")
        X, y = make_classification(
            n_samples=params['n_samples'],
            n_features=params['n_features'],
            n_informative=15,
            n_redundant=5,
            random_state=params['random_state']
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=params['random_state']
        )
        
        # Train model
        print("Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            random_state=params['random_state'],
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
        }
        
        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        print(f"Model accuracy: {metrics['accuracy']:.4f}")
        print(f"Model precision: {metrics['precision']:.4f}")
        print(f"Model recall: {metrics['recall']:.4f}")
        print(f"Model F1: {metrics['f1_score']:.4f}")
        
        # Log model to MLflow
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="churn-prediction-model"
        )
        print("Model logged to MLflow")
        
        # Also save locally (optional)
        model_path = '/models/model.joblib'
        os.makedirs('/models', exist_ok=True)
        import joblib
        joblib.dump(model, model_path)
        print(f"Model also saved locally to {model_path}")
        
        print(f"Training completed at {datetime.now()}")
        return metrics['accuracy']

if __name__ == "__main__":
    try:
        accuracy = train_model()
        print(f"Final accuracy: {accuracy:.4f}")
    except Exception as e:
        print(f"Error during training: {e}")
        raise
