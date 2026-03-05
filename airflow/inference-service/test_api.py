import os
import json
import requests
import pandas as pd
from pathlib import Path

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
TRAINING_CSV = DATA_DIR / "customer_churn_dataset-training-master.csv"
TESTING_CSV = DATA_DIR / "customer_churn_dataset-testing-master.csv"

# Columns to drop when sending to API (target and ID-like)
CHURN_COL = "churn"
ID_PATTERNS = ["id", "customer", "customername", "name", "customerid"]

# Defaults for /predict when CSV has missing values (required by PredictionInput)
# Based on actual ChurnGuard CSV columns
PREDICT_DEFAULTS = {
    "age": 0,
    "gender": "Unknown",
    "tenure": 0,
    "usage_frequency": 0,
    "support_calls": 0,
    "payment_delay": 0,
    "subscription_type": "Basic",
    "contract_length": "Monthly",
    "total_spend": 0.0,
    "last_interaction": 0,
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to match ChurnGuard/train schema (lowercase, underscores)."""
    df = df.copy()
    df.columns = df.columns.astype(str).str.lower().str.strip().str.replace(" ", "_")
    return df


def _row_to_payload(row: pd.Series, drop_churn_and_id: bool = True) -> dict:
    """Convert a DataFrame row to API payload dict. Drops churn and ID columns."""
    out = row.to_dict()
    if drop_churn_and_id:
        keys_to_drop = [k for k in out if k == CHURN_COL or any(p in k for p in ID_PATTERNS)]
        for k in keys_to_drop:
            out.pop(k, None)
    # Coerce for JSON: handle NaN, timestamps, empty strings; numeric cols to int/float
    numeric_float = {"total_spend"}
    numeric_int = {"age", "tenure", "usage_frequency", "support_calls", "payment_delay", "last_interaction"}
    for k, v in list(out.items()):
        if pd.isna(v) or (isinstance(v, str) and v.strip() == ""):
            out[k] = None
        elif k in numeric_float:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = 0.0
        elif k in numeric_int:
            try:
                out[k] = int(float(v))
            except (TypeError, ValueError):
                out[k] = 0
        elif isinstance(v, (pd.Timestamp,)):
            out[k] = str(v)
        elif hasattr(v, "item"):
            out[k] = v.item()
        elif isinstance(v, str):
            out[k] = v.strip() if v else None
    return out


def load_churnguard_records(max_single: int = 1, max_batch: int = 5):
    """
    Load records from ChurnGuard CSVs for API testing.
    Returns (single_record_or_none, list_for_batch) so we can use real data in tests.
    """
    single = None
    batch = []
    for path in [TRAINING_CSV, TESTING_CSV]:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            df = _normalize_columns(df)
            if CHURN_COL not in df.columns:
                continue
            # Drop rows with missing churn for cleaner payloads
            df = df.dropna(subset=[CHURN_COL])
            for _, row in df.iterrows():
                payload = _row_to_payload(row)
                if single is None and payload:
                    single = payload
                if len(batch) < max_batch and payload:
                    batch.append(payload)
                if single is not None and len(batch) >= max_batch:
                    return single, batch
        except Exception as e:
            print(f"Warning: could not load {path}: {e}")
    return single, batch


def test_health():
    """Test health endpoint"""
    print("\n=== Testing Health Endpoint ===")
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_model_info():
    """Test model info endpoint"""
    print("\n=== Testing Model Info Endpoint ===")
    response = requests.get(f"{API_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def _payload_for_predict(payload: dict) -> dict:
    """Ensure payload has all required fields for /predict; fill missing with defaults."""
    out = {**PREDICT_DEFAULTS, **{k: v for k, v in payload.items() if v is not None}}
    return out


def test_single_prediction():
    """Test single prediction using ChurnGuard data or fallback sample."""
    print("\n=== Testing Single Prediction ===")
    single, _ = load_churnguard_records(max_single=1, max_batch=1)
    if single is None:
        print("No ChurnGuard CSV found; using sample record.")
        single = PREDICT_DEFAULTS.copy()
        single.update({
            "age": 35,
            "gender": "Male",
            "tenure": 24,
            "usage_frequency": 15,
            "support_calls": 3,
            "payment_delay": 10,
            "subscription_type": "Premium",
            "contract_length": "Annual",
            "total_spend": 500.0,
            "last_interaction": 5,
        })
    else:
        print("Using record from ChurnGuard CSV.")
        single = _payload_for_predict(single)
    response = requests.post(
        f"{API_URL}/predict",
        json=single,
        headers={"Content-Type": "application/json"},
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_single_prediction_from_record():
    """Test /predict/from-record with a raw ChurnGuard CSV row."""
    print("\n=== Testing Single Prediction (from-record) ===")
    single, _ = load_churnguard_records(max_single=1, max_batch=1)
    if single is None:
        print("No ChurnGuard CSV found; skipping from-record test.")
        return True
    response = requests.post(
        f"{API_URL}/predict/from-record",
        json=single,
        headers={"Content-Type": "application/json"},
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_batch_prediction():
    """Test batch prediction using ChurnGuard data or fallback sample."""
    print("\n=== Testing Batch Prediction ===")
    _, batch = load_churnguard_records(max_single=0, max_batch=5)
    if not batch:
        print("No ChurnGuard CSV found; using sample batch.")
        batch = [
            _payload_for_predict({
                "age": 35, "gender": "Male", "tenure": 24, "usage_frequency": 15,
                "support_calls": 3, "payment_delay": 10, "subscription_type": "Premium",
                "contract_length": "Annual", "total_spend": 500.0, "last_interaction": 5,
            }),
            _payload_for_predict({
                "age": 45, "gender": "Female", "tenure": 48, "usage_frequency": 25,
                "support_calls": 1, "payment_delay": 0, "subscription_type": "Standard",
                "contract_length": "Annual", "total_spend": 1200.0, "last_interaction": 2,
            }),
        ]
    else:
        print(f"Using {len(batch)} records from ChurnGuard CSV.")
        batch = [_payload_for_predict(b) for b in batch]
    data = {"instances": batch}
    response = requests.post(
        f"{API_URL}/predict/batch",
        json=data,
        headers={"Content-Type": "application/json"},
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_root():
    """Test root endpoint"""
    print("\n=== Testing Root Endpoint ===")
    response = requests.get(f"{API_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def main():
    """Run all tests"""
    print(f"Testing API at: {API_URL}")
    print("=" * 50)
    tests = [
        ("Root", test_root),
        ("Health", test_health),
        ("Model Info", test_model_info),
        ("Single Prediction", test_single_prediction),
        ("Single Prediction (from-record)", test_single_prediction_from_record),
        ("Batch Prediction", test_batch_prediction),
    ]
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} failed with error: {str(e)}")
            results[test_name] = False
    print("\n" + "=" * 50)
    print("=== Test Summary ===")
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")


if __name__ == "__main__":
    main()
