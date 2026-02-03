"""
ChurnGuard ML Model Training Script
Production-ready customer churn prediction model with MLflow Integration
"""
import os
import zipfile
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import boto3
from botocore.exceptions import ClientError
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import mlflow
import mlflow.sklearn
import joblib
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent
TEMP_DATA_DIR = SCRIPT_DIR / "temp_data"
MODELS_DIR = SCRIPT_DIR / "models"

# S3 bucket configuration (same AWS account)
S3_BUCKET_NAME = "test-skip-env"  # e.g. "my-datasets-bucket"
S3_DATASET_OBJECT = "cf-test/customer-churn-dataset.zip"  # object key in the bucket
S3_REGION = "us-east-1"  # optional, e.g. "us-east-1"


def download_from_s3(bucket_name, object_key, local_path, region_name=None):
    """
    Download file from AWS S3 bucket using default credentials (same AWS account).
    Uses IAM role, env vars (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY), or ~/.aws/credentials.
    """
    try:
        logger.info(f"Connecting to S3 bucket: {bucket_name}")
        logger.debug(f"Object: {object_key}, Region: {region_name or 'default'}")

        # Standard S3 client - uses default credential chain (same account)
        client_kwargs = {'service_name': 's3'}
        if region_name:
            client_kwargs['region_name'] = region_name
        s3_client = boto3.client(**client_kwargs)

        # Check if bucket exists
        s3_client.head_bucket(Bucket=bucket_name)
        logger.debug(f"Bucket '{bucket_name}' accessible")

        # Download file
        logger.info(f"Downloading {object_key}...")
        s3_client.download_file(bucket_name, object_key, local_path)

        file_size = os.path.getsize(local_path) / (1024 * 1024)  # MB
        logger.info(f"Downloaded successfully ({file_size:.2f} MB)")
        return True

    except ClientError as e:
        logger.error(f"S3 download failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False


def download_and_extract_dataset(data_dir=None, force_download=True):
    """
    Download and extract the customer churn dataset from AWS S3 (same account).
    Always downloads fresh data from S3 unless force_download=False.
    """
    # Use absolute path relative to script location
    if data_dir is None:
        data_dir = TEMP_DATA_DIR
    else:
        data_dir = Path(data_dir).resolve()
    
    logger.info(f"Starting dataset download from AWS S3 to {data_dir}")

    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    bucket_name = S3_BUCKET_NAME
    dataset_object = S3_DATASET_OBJECT
    region_name = S3_REGION

    zip_path = os.path.join(data_dir, os.path.basename(dataset_object))
    train_file = os.path.join(data_dir, 'customer_churn_dataset-training-master.csv')
    test_file = os.path.join(data_dir, 'customer_churn_dataset-testing-master.csv')

    # Check if files already exist (skip download only if force_download=False)
    if not force_download and os.path.exists(train_file) and os.path.exists(test_file):
        logger.info(f"Dataset files already exist in {data_dir} (using cached version)")
        return train_file, test_file

    # Always download fresh data from S3
    if not bucket_name:
        logger.error("S3_BUCKET_NAME is empty. Cannot download dataset.")
        return None, None
    
    logger.info("Downloading fresh dataset from S3...")

    # Remove old files if they exist
    if os.path.exists(zip_path):
        os.remove(zip_path)
        logger.debug(f"Removed old zip file: {zip_path}")
    if os.path.exists(train_file):
        os.remove(train_file)
        logger.debug(f"Removed old training file")
    if os.path.exists(test_file):
        os.remove(test_file)
        logger.debug(f"Removed old testing file")

    success = download_from_s3(
        bucket_name=bucket_name,
        object_key=dataset_object,
        local_path=zip_path,
        region_name=region_name
    )

    if not success:
        logger.error(f"S3 download failed. Bucket: {bucket_name}, Object: {dataset_object}")
        logger.info("To proceed, place CSVs manually in temp_data/ or fix S3 config.")
        return None, None

    # Extract zip file
    logger.info(f"Extracting {os.path.basename(zip_path)}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Verify expected files exist
        if os.path.exists(train_file) and os.path.exists(test_file):
            logger.info("Dataset extraction completed successfully")
            return train_file, test_file
        else:
            logger.error(f"Expected CSV files not found after extraction")
            return None, None
            
    except zipfile.BadZipFile:
        logger.error(f"Invalid zip file: {zip_path}")
        return None, None
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return None, None


def load_data(data_path):
    """
    Load ChurnGuard data from CSV file.
    Expects CSV with customer churn features.
    """
    if not data_path or not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    logger.debug(f"Dataset shape: {df.shape}")
    logger.debug(f"Columns: {list(df.columns)[:5]}...")  # Show first 5 columns
    
    # Normalize column names (lowercase, strip spaces, replace spaces with underscores)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Common target column names
    target_aliases = ['churn', 'churned', 'exited', 'attrition', 'customer_status', 'status']
    target_col = None
    for alias in target_aliases:
        if alias in df.columns:
            target_col = alias
            break
    
    if target_col and target_col != 'churn':
        logger.info(f"Renaming target column '{target_col}' to 'churn'")
        df = df.rename(columns={target_col: 'churn'})
    
    # Convert churn to binary if needed
    if 'churn' in df.columns:
        if df['churn'].dtype == 'object':
            # Map common values (strip whitespace for robustness)
            churn_map = {
                'Yes': 1, 'No': 0,
                'yes': 1, 'no': 0,
                'TRUE': 1, 'FALSE': 0,
                'True': 1, 'False': 0,
                'Churned': 1, 'Stayed': 0,
                'Exited': 1, 'Retained': 0,
                '1': 1, '0': 0
            }
            df['churn'] = df['churn'].astype(str).str.strip().replace(churn_map)
            # Coerce to numeric; invalid/empty become NaN
            df['churn'] = pd.to_numeric(df['churn'], errors='coerce')
        # Drop rows with missing target (sklearn does not accept NaN in y)
        before = len(df)
        df = df.dropna(subset=['churn'])
        df['churn'] = df['churn'].astype(int)
        dropped = before - len(df)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows with missing/invalid churn label")
        
        churn_rate = df['churn'].mean() * 100
        logger.info(f"Churn rate: {churn_rate:.2f}% ({df['churn'].sum()}/{len(df)} samples)")
    
    return df


def preprocess_data(df):
    """Preprocess and engineer features for churn prediction - FLEXIBLE for any dataset"""
    
    # Create a copy
    df = df.copy()
    
    logger.info("Preprocessing data")
    
    # Check if churn column exists
    if 'churn' not in df.columns:
        raise ValueError("Dataset must have a 'churn' column as the target variable")
    
    # Separate target and drop rows with NaN in churn (sklearn rejects NaN in y)
    y = df['churn'].copy()
    X = df.drop('churn', axis=1)
    nan_mask = y.isna()
    if nan_mask.any():
        n_drop = nan_mask.sum()
        logger.warning(f"Dropping {n_drop} rows with missing churn target")
        X = X.loc[~nan_mask].copy()
        y = y.loc[~nan_mask].astype(int)
    else:
        y = y.astype(int)
    
    # Remove ID columns (usually non-predictive)
    id_patterns = ['id', 'customer', 'customername', 'name', 'customerid']
    cols_to_drop = []
    for col in X.columns:
        if any(pattern in col.lower() for pattern in id_patterns):
            cols_to_drop.append(col)
    
    if cols_to_drop:
        logger.debug(f"Dropping ID columns: {cols_to_drop}")
        X = X.drop(cols_to_drop, axis=1)
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    logger.info(f"Features: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
    
    # Handle missing values in numeric columns
    if X[numeric_features].isnull().sum().sum() > 0:
        n_missing = X[numeric_features].isnull().sum().sum()
        logger.debug(f"Filling {n_missing} missing numeric values with median")
        for col in numeric_features:
            X[col] = X[col].fillna(X[col].median())
    
    # Handle missing values in categorical columns
    if X[categorical_features].isnull().sum().sum() > 0:
        n_missing = X[categorical_features].isnull().sum().sum()
        logger.debug(f"Filling {n_missing} missing categorical values with mode")
        for col in categorical_features:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
    
    # Feature Engineering (generic approach - no hardcoded features)
    # If you have specific domain features, add them here
    # For now, we'll work with the raw features as-is
    
    logger.info(f"Final feature count: {len(X.columns)}")
    
    return X, y, categorical_features, numeric_features


def cleanup_downloaded_files(data_dir=None):
    """
    Clean up downloaded dataset files after training.
    Removes CSV files, zip file, and the entire temp directory.
    """
    # Use absolute path relative to script location
    if data_dir is None:
        data_dir = TEMP_DATA_DIR
    else:
        data_dir = Path(data_dir).resolve()
    
    logger.info(f"Cleaning up downloaded dataset files from {data_dir}...")
    
    # Try to remove the entire directory
    if os.path.exists(data_dir):
        try:
            shutil.rmtree(data_dir)
            logger.info(f"Removed entire directory: {data_dir}")
            return
        except Exception as e:
            logger.warning(f"Failed to remove directory {data_dir}: {e}")
            # Fall back to removing individual files
    
    # Fallback: remove individual files if directory removal failed
    files_to_remove = [
        os.path.join(data_dir, 'customer_churn_dataset-training-master.csv'),
        os.path.join(data_dir, 'customer_churn_dataset-testing-master.csv'),
        os.path.join(data_dir, os.path.basename(S3_DATASET_OBJECT))
    ]
    
    removed_count = 0
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.debug(f"Removed: {os.path.basename(file_path)}")
                removed_count += 1
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {e}")
    
    if removed_count > 0:
        logger.info(f"Cleaned up {removed_count} file(s)")
    
    # Try to remove empty directory
    if os.path.exists(data_dir):
        try:
            os.rmdir(data_dir)
            logger.debug(f"Removed empty directory: {data_dir}")
        except Exception as e:
            logger.debug(f"Could not remove directory (may not be empty): {e}")


def train_model():
    """Train production-ready churn prediction model"""
    logger.info(f"Starting ChurnGuard model training at {datetime.now()}")
    
    # Set MLflow configuration
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    mlflow.set_tracking_uri(mlflow_uri)
    logger.info(f"MLflow tracking URI: {mlflow_uri}")
    
    os.environ['MLFLOW_TRACKING_USERNAME'] = "user"
    os.environ['MLFLOW_TRACKING_PASSWORD'] = "JWu4h4SK0BNW"

    experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'churn-prediction')
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment: {experiment_name}")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"churnguard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Download and extract dataset from AWS S3
        train_file, test_file = download_and_extract_dataset()
        
        if train_file is None or test_file is None:
            # S3 download failed - raise error instead of falling back to dummy data
            logger.error("ChurnGuard dataset not available")
            logger.error(f"S3_BUCKET_NAME = '{S3_BUCKET_NAME}'")
            logger.error(f"S3_DATASET_OBJECT = '{S3_DATASET_OBJECT}'")
            logger.info("Place CSVs manually in temp_data/ or fix S3 config:")
            logger.info("  - temp_data/customer_churn_dataset-training-master.csv")
            logger.info("  - temp_data/customer_churn_dataset-testing-master.csv")
            raise RuntimeError("ChurnGuard dataset not available. Cannot train model.")
        
        # Use separate training and testing files
        logger.info("Loading training and testing datasets")
        
        # Load training data
        train_df = load_data(train_file)
        X_train, y_train, categorical_features, numeric_features = preprocess_data(train_df)
        
        # Load testing data
        test_df = load_data(test_file)
        X_test, y_test, _, _ = preprocess_data(test_df)
        
        # Ensure test data has same columns as training data
        for col in X_train.columns:
            if col not in X_test.columns:
                X_test[col] = 0
        X_test = X_test[X_train.columns]
        
        logger.info(f"Training set: {len(X_train)} samples, churn rate: {y_train.mean()*100:.1f}%")
        logger.info(f"Test set: {len(X_test)} samples, churn rate: {y_test.mean()*100:.1f}%")
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
                 categorical_features)
            ])
        
        # Model parameters
        params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'random_state': 42,
            'class_weight': 'balanced',
        }
        mlflow.log_params(params)
        
        # Create full pipeline
        logger.info("Training Random Forest model...")
        model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(**params, n_jobs=-1))
        ])
        
        # Train model
        model_pipeline.fit(X_train, y_train)
        logger.info("Model training completed")
        
        # Make predictions
        y_pred = model_pipeline.predict(X_test)
        y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
        }
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Print metrics
        logger.info("="*50)
        logger.info("MODEL PERFORMANCE METRICS")
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1 Score:  {metrics['f1_score']:.4f}")
        logger.info(f"ROC AUC:   {metrics['roc_auc']:.4f}")
        logger.info("="*50)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
        
        # Feature importance
        feature_names_cat = model_pipeline.named_steps['preprocessor']\
            .named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_feature_names = numeric_features + list(feature_names_cat)
        
        feature_importance = pd.DataFrame({
            'feature': all_feature_names,
            'importance': model_pipeline.named_steps['classifier'].feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        logger.info("Top 10 Important Features:")
        for _, row in feature_importance.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Log feature importance as artifact
        feature_importance.to_csv('feature_importance.csv', index=False)
        mlflow.log_artifact('feature_importance.csv')
        os.remove('feature_importance.csv')
        
        # Log model to MLflow
        logger.info("Logging model to MLflow Model Registry...")
        mlflow.sklearn.log_model(
            model_pipeline,
            "model",
            registered_model_name="churn-prediction-model",
            input_example=X_train.head(1),
        )
        logger.info("Model logged to MLflow Model Registry")
        
        # Save locally (optional, in project directory)
        model_dir = MODELS_DIR
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "churnguard_model.joblib"
        joblib.dump(model_pipeline, model_path)
        logger.info(f"Model saved locally to {model_path}")
        
        # Clean up downloaded dataset files
        cleanup_downloaded_files()
        
        logger.info(f"Training completed at {datetime.now()}")
        return metrics['roc_auc']

if __name__ == "__main__":
    try:
        roc_auc = train_model()
        logger.info("="*50)
        logger.info(f"TRAINING SUCCESS - ROC AUC: {roc_auc:.4f}")
        logger.info("="*50)
    except Exception as e:
        logger.error("="*50)
        logger.error(f"ERROR during training: {e}")
        logger.error("="*50)
        # Clean up files even on failure
        try:
            cleanup_downloaded_files()
        except:
            pass
        raise
