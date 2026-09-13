import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

def generate_shap_analysis(csv_path: str, target_col: str) -> dict:
    """Génère un graphique SHAP (Beeswarm plot) pour expliquer l'impact de chaque variable."""
    df = pd.read_csv(csv_path)
    ignore_cols = [c for c in df.columns if 'id' in c.lower() and c.lower() != target_col.lower()]
    X = df.drop(columns=[target_col] + ignore_cols)
    y = df[target_col]

    is_classif = (y.dtype == 'object' or y.dtype == 'bool' or y.nunique() <= 10)

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'string', 'bool']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
        ]
    )

    X_processed = preprocessor.fit_transform(X)
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = list(cat_encoder.get_feature_names_out(cat_cols)) if cat_cols else []
    feature_names = num_cols + cat_feature_names

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    if is_classif:
        if y.dtype == 'object':
            y_train = y_train.astype('category').cat.codes
        model = XGBClassifier(eval_metric='logloss', random_state=42)
    else:
        model = XGBRegressor(random_state=42)

    model.fit(X_train, y_train)

    # Calcul des valeurs SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # Génération du graphique SHAP Summary
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title("Interprétabilité SHAP - Impact des variables sur les prédictions", fontsize=12, pad=15)
    plt.tight_layout()
    
    shap_path = os.path.join(CHARTS_DIR, "shap_summary.png")
    plt.savefig(shap_path, dpi=200, bbox_inches='tight')
    plt.close()

    return {"status": "success", "shap_chart": shap_path}