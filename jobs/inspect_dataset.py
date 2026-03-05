"""
Dataset Inspector
Quick script to inspect your churn dataset and see what columns it has
"""
import pandas as pd
import sys

def inspect_dataset(file_path):
    """Inspect and display dataset information"""
    
    print("="*60)
    print("DATASET INSPECTOR")
    print("="*60)
    
    try:
        # Load dataset
        print(f"\nLoading dataset from: {file_path}")
        df = pd.read_csv(file_path)
        
        # Basic info
        print(f"\n✓ Dataset loaded successfully!")
        print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Column info
        print("\n" + "-"*60)
        print("COLUMNS")
        print("-"*60)
        for i, col in enumerate(df.columns, 1):
            dtype = df[col].dtype
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            print(f"{i:2d}. {col:30s} | {str(dtype):10s} | {unique_count:4d} unique | {null_count:4d} nulls")
        
        # Target column detection
        print("\n" + "-"*60)
        print("TARGET COLUMN DETECTION")
        print("-"*60)
        target_aliases = ['churn', 'churned', 'exited', 'attrition', 'customer_status', 'status']
        found_target = None
        for alias in target_aliases:
            matching_cols = [col for col in df.columns if alias.lower() in col.lower()]
            if matching_cols:
                found_target = matching_cols[0]
                print(f"✓ Found potential target column: '{found_target}'")
                print(f"  Values: {df[found_target].unique()}")
                print(f"  Distribution: {dict(df[found_target].value_counts())}")
                break
        
        if not found_target:
            print("⚠ No target column found. Looking for binary columns...")
            binary_cols = [col for col in df.columns if df[col].nunique() == 2]
            if binary_cols:
                print(f"Potential target columns (binary): {binary_cols}")
        
        # Data types summary
        print("\n" + "-"*60)
        print("DATA TYPES SUMMARY")
        print("-"*60)
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        print(f"Numeric columns ({len(numeric_cols)}):")
        for col in numeric_cols:
            print(f"  - {col}")
        print(f"\nCategorical columns ({len(categorical_cols)}):")
        for col in categorical_cols:
            print(f"  - {col}")
        
        # First few rows
        print("\n" + "-"*60)
        print("SAMPLE DATA (first 3 rows)")
        print("-"*60)
        print(df.head(3).to_string())
        
        # Missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print("\n" + "-"*60)
            print("MISSING VALUES")
            print("-"*60)
            missing_df = pd.DataFrame({
                'Column': missing[missing > 0].index,
                'Missing Count': missing[missing > 0].values,
                'Percentage': (missing[missing > 0].values / len(df) * 100).round(2)
            })
            print(missing_df.to_string(index=False))
        else:
            print("\n✓ No missing values found!")
        
        # Summary stats for numeric columns
        print("\n" + "-"*60)
        print("NUMERIC COLUMNS SUMMARY")
        print("-"*60)
        print(df[numeric_cols].describe().round(2).to_string())
        
        print("\n" + "="*60)
        print("✓ Inspection complete!")
        print("="*60)
        
    except FileNotFoundError:
        print(f"\n❌ Error: File not found at '{file_path}'")
        print("Please check the file path and try again.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Default path
        file_path = input("Enter path to your churn dataset CSV: ")
    
    inspect_dataset(file_path)
