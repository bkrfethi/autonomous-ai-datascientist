import pandas as pd
import json

def profile_dataset(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    
    profile = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": {},
        "missing_values_total": int(df.isnull().sum().sum())
    }
    
    for col in df.columns:
        profile["columns"][col] = {
            "type": str(df[col].dtype),
            "missing_values": int(df[col].isnull().sum()),
            "unique_values": int(df[col].nunique()),
            "sample_values": df[col].dropna().unique()[:3].tolist()
        }
        
    return profile

if __name__ == "__main__":
    res = profile_dataset("churn_data.csv")
    print(json.dumps(res, indent=2))