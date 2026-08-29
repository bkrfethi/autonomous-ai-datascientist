import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import json

def train_and_evaluate(csv_path: str, target_col: str) -> dict:
    """Entraîne et compare des modèles ML, puis extrait l'importance des variables."""
    df = pd.read_csv(csv_path)
    
    ignore_cols = [c for c in df.columns if 'id' in c.lower()]
    X = df.drop(columns=[target_col] + ignore_cols)
    y = df[target_col]


    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

    # Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    X_processed = preprocessor.fit_transform(X)
    
    # Récupération des noms de colonnes après One-Hot Encoding
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
    feature_names = num_cols + list(cat_feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. Baseline
    baseline = LogisticRegression()
    baseline.fit(X_train, y_train)
    y_pred_base = baseline.predict(X_test)
    y_prob_base = baseline.predict_proba(X_test)[:, 1]

    # 2. XGBoost
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

    # 3. Extraction de l'importance des variables (XGBoost)
    importances = xgb.feature_importances_
    feature_importance_dict = dict(zip(feature_names, [float(x) for x in importances]))
    sorted_importances = dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True))

    results = {
        "models": {
            "LogisticRegression": {
                "accuracy": round(float(accuracy_score(y_test, y_pred_base)), 4),
                "f1_score": round(float(f1_score(y_test, y_pred_base)), 4),
                "roc_auc": round(float(roc_auc_score(y_test, y_prob_base)), 4)
            },
            "XGBoost": {
                "accuracy": round(float(accuracy_score(y_test, y_pred_xgb)), 4),
                "f1_score": round(float(f1_score(y_test, y_pred_xgb)), 4),
                "roc_auc": round(float(roc_auc_score(y_test, y_prob_xgb)), 4)
            }
        },
        "top_features": sorted_importances
    }
    
    return results

if __name__ == "__main__":
    res = train_and_evaluate("churn_data.csv", target_col="churn")
    print(json.dumps(res, indent=2))