import pandas as pd
import numpy as np
from scipy.stats import skew
import json

def analyze_dataset(csv_path: str, target_col: str = None) -> dict:
    """Analyse en profondeur la structure, la qualité et les risques d'un dataset."""
    df = pd.read_csv(csv_path)
    
    n_rows, n_cols = df.shape
    memory_usage_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    
    col_types = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "id_or_high_cardinality": []
    }
    
    warnings = []
    
    for col in df.columns:
        if col.lower() == target_col.lower() if target_col else False:
            continue
            
        unique_count = df[col].nunique()
        dtype = df[col].dtype
        
        if 'id' in col.lower() or unique_count == n_rows:
            col_types["id_or_high_cardinality"].append(col)
            warnings.append(f"La colonne '{col}' ressemble à un identifiant (unique à 100%). Elle sera exclue de l'entraînement.")
            continue
            
        if pd.api.types.is_numeric_dtype(dtype):
            if unique_count <= 10:
                col_types["categorical"].append(col)
            else:
                col_types["numeric"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            col_types["datetime"].append(col)
        else:
            if unique_count > 50 and unique_count / n_rows > 0.3:
                col_types["id_or_high_cardinality"].append(col)
                warnings.append(f"La colonne textuelle '{col}' a une très forte cardinalité ({unique_count} valeurs uniques).")
            else:
                col_types["categorical"].append(col)

    missing_pct = (df.isnull().sum() / n_rows * 100).to_dict()
    missing_warnings = {k: round(v, 2) for k, v in missing_pct.items() if v > 0}
    if missing_warnings:
        warnings.append(f"Valeurs manquantes détectées : {missing_warnings}")

    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        warnings.append(f"Colonnes constantes sans variance détectées : {constant_cols}")

    target_info = {}
    if target_col and target_col in df.columns:
        y = df[target_col]
        is_classif = (y.dtype == 'object' or y.dtype == 'bool' or y.nunique() <= 10)
        
        target_info["type"] = "Classification" if is_classif else "Regression"
        
        if is_classif:
            dist = y.value_counts(normalize=True).to_dict()
            target_info["distribution"] = {str(k): round(v, 4) for k, v in dist.items()}
            min_class_pct = min(dist.values())
            if min_class_pct < 0.15:
                warnings.append(f"Forte imbalancement de la variable cible ({round(min_class_pct*100, 1)}% pour la classe minoritaire). Poids de classes ou rééchantillonnage requis.")
        else:
            target_info["mean"] = round(float(y.mean()), 2)
            target_info["std"] = round(float(y.std()), 2)
            target_skew = skew(y.dropna())
            target_info["skewness"] = round(float(target_skew), 2)
            if abs(target_skew) > 1.5:
                warnings.append(f"Forte asymétrie (skewness = {round(target_skew, 2)}) sur la cible. Une transformation logarithmique est conseillée.")

        numeric_df = df.select_dtypes(include=[np.number])
        if target_col in numeric_df.columns:
            correlations = numeric_df.corr()[target_col].abs()
            for col, corr in correlations.items():
                if col != target_col and corr > 0.90:
                    warnings.append(f"Risque de Target Leakage : La colonne '{col}' a une corrélation de {round(corr, 2)} avec la cible.")

    return {
        "dataset_summary": {
            "rows": n_rows,
            "columns": n_cols,
            "memory_mb": memory_usage_mb
        },
        "column_types": col_types,
        "target_analysis": target_info,
        "data_quality_warnings": warnings
    }

if __name__ == "__main__":
    res = analyze_dataset("uploads/churn_data.csv", target_col="churn")
    print(json.dumps(res, indent=2, ensure_ascii=False))