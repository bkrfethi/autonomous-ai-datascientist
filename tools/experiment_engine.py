import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_absolute_error, root_mean_squared_error
)

from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb
import catboost as cb

def run_experiment(
    csv_path: str,
    target_col: str,
    model_name: str = "xgboost",
    handle_imbalance: bool = False,
    log_transform_target: bool = False
) -> dict:
    """Exécute une expérience ML paramétrée et enregistre ses résultats."""
    df = pd.read_csv(csv_path)

    ignore_cols = [c for c in df.columns if 'id' in c.lower() and c.lower() != target_col.lower()]
    X = df.drop(columns=[target_col] + ignore_cols)
    y = df[target_col]

    is_classif = (y.dtype == 'object' or y.dtype == 'bool' or y.nunique() <= 10)
    task_type = "Classification" if is_classif else "Regression"

    if not is_classif and log_transform_target:
        y = np.log1p(y)

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'string', 'bool']).columns.tolist()

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ]
    )

    X_processed = preprocessor.fit_transform(X)
    
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = list(cat_encoder.get_feature_names_out(cat_cols)) if cat_cols else []
    feature_names = num_cols + cat_feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    m_name = model_name.lower()
    
    if is_classif:
        if y.dtype == 'object':
            y_train = y_train.astype('category').cat.codes
            y_test = y_test.astype('category').cat.codes

        scale_pos_weight = 1.0
        if handle_imbalance and len(np.bincount(y_train)) == 2:
            neg, pos = np.bincount(y_train)
            scale_pos_weight = neg / pos if pos > 0 else 1.0

        if m_name == "logistic":
            model = LogisticRegression(class_weight='balanced' if handle_imbalance else None, max_iter=1000)
        elif m_name == "lightgbm":
            model = lgb.LGBMClassifier(random_state=42, verbose=-1, scale_pos_weight=scale_pos_weight)
        elif m_name == "catboost":
            model = cb.CatBoostClassifier(random_state=42, verbose=0, scale_pos_weight=scale_pos_weight)
        else:
            model = XGBClassifier(eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos_weight)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            roc_auc = round(float(roc_auc_score(y_test, y_prob)), 4)
        except Exception:
            roc_auc = None

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred, average='weighted')), 4),
            "roc_auc": roc_auc
        }

    else:
        if m_name == "ridge":
            model = Ridge()
        elif m_name == "lightgbm":
            model = lgb.LGBMRegressor(random_state=42, verbose=-1)
        elif m_name == "catboost":
            model = cb.CatBoostRegressor(random_state=42, verbose=0)
        else:
            model = XGBRegressor(random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if log_transform_target:
            y_test_orig = np.expm1(y_test)
            y_pred_orig = np.expm1(y_pred)
        else:
            y_test_orig = y_test
            y_pred_orig = y_pred

        metrics = {
            "r2_score": round(float(r2_score(y_test_orig, y_pred_orig)), 4),
            "mae": round(float(mean_absolute_error(y_test_orig, y_pred_orig)), 2),
            "rmse": round(float(root_mean_squared_error(y_test_orig, y_pred_orig)), 2)
        }

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0] if model.coef_.ndim > 1 else model.coef_)
    else:
        importances = np.zeros(len(feature_names))

    sorted_importances = dict(sorted(
        zip(feature_names, [float(x) for x in importances]),
        key=lambda x: x[1], reverse=True
    ))

    return {
        "experiment_config": {
            "model_name": model_name,
            "task_type": task_type,
            "handle_imbalance": handle_imbalance,
            "log_transform_target": log_transform_target
        },
        "metrics": metrics,
        "top_features": dict(list(sorted_importances.items())[:10])
    }

if __name__ == "__main__":
    res = run_experiment("uploads/churn_data.csv", target_col="churn", model_name="catboost", handle_imbalance=True)
    print(json.dumps(res, indent=2))