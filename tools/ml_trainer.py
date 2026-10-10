import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_absolute_error, root_mean_squared_error
from xgboost import XGBClassifier, XGBRegressor
import json

def train_and_evaluate(csv_path: str, target_col: str) -> dict:
    df = pd.read_csv(csv_path)
    
    ignore_cols = [c for c in df.columns if 'id' in c.lower() and c.lower() != target_col.lower()]
    X = df.drop(columns=[target_col] + ignore_cols)
    y = df[target_col]

    is_classification = False
    if y.dtype == 'object' or y.dtype == 'bool' or y.nunique() <= 10:
        is_classification = True

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
    cat_feature_names = cat_encoder.get_feature_names_out(cat_cols) if cat_cols else []
    feature_names = num_cols + list(cat_feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    if is_classification:
        if y.dtype == 'object':
            y_train = y_train.astype('category').cat.codes
            y_test = y_test.astype('category').cat.codes

        baseline = LogisticRegression()
        baseline.fit(X_train, y_train)
        y_pred_base = baseline.predict(X_test)

        xgb = XGBClassifier(eval_metric='logloss', random_state=42)
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)

        models_perf = {
            "LogisticRegression": {
                "accuracy": round(float(accuracy_score(y_test, y_pred_base)), 4),
                "f1_score": round(float(f1_score(y_test, y_pred_base, average='weighted')), 4)
            },
            "XGBoost": {
                "accuracy": round(float(accuracy_score(y_test, y_pred_xgb)), 4),
                "f1_score": round(float(f1_score(y_test, y_pred_xgb, average='weighted')), 4)
            }
        }
    else:
        baseline = Ridge()
        baseline.fit(X_train, y_train)
        y_pred_base = baseline.predict(X_test)

        xgb = XGBRegressor(random_state=42)
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)

        models_perf = {
            "RidgeRegression": {
                "r2_score": round(float(r2_score(y_test, y_pred_base)), 4),
                "mae": round(float(mean_absolute_error(y_test, y_pred_base)), 2),
                "rmse": round(float(root_mean_squared_error(y_test, y_pred_base)), 2)
            },
            "XGBoost": {
                "r2_score": round(float(r2_score(y_test, y_pred_xgb)), 4),
                "mae": round(float(mean_absolute_error(y_test, y_pred_xgb)), 2),
                "rmse": round(float(root_mean_squared_error(y_test, y_pred_xgb)), 2)
            }
        }

    importances = xgb.feature_importances_
    sorted_importances = dict(sorted(
        zip(feature_names, [float(x) for x in importances]),
        key=lambda x: x[1], reverse=True
    ))

    return {
        "task_type": "Classification" if is_classification else "Regression",
        "models": models_perf,
        "top_features": sorted_importances
    }