import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier, XGBRegressor

CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

def generate_charts(csv_path: str, target_col: str) -> dict:
    """Génère des graphiques analytiques et les sauvegarde au format PNG."""
    sns.set_theme(style="whitegrid")
    
    df = pd.read_csv(csv_path)
    ignore_cols = [c for c in df.columns if 'id' in c.lower() and c.lower() != target_col.lower()]
    X = df.drop(columns=[target_col] + ignore_cols)
    y = df[target_col]

    is_classification = (y.dtype == 'object' or y.dtype == 'bool' or y.nunique() <= 10)

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

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    chart_paths = []

    if is_classification:
        if y.dtype == 'object':
            y_train = y_train.astype('category').cat.codes
            y_test = y_test.astype('category').cat.codes
        model = XGBClassifier(eval_metric='logloss', random_state=42)
    else:
        model = XGBRegressor(random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(10)

    plt.figure(figsize=(8, 4))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='crest')
    plt.title('Top 10 - Importance des Variables')
    plt.tight_layout()
    fi_path = os.path.join(CHARTS_DIR, 'feature_importance.png')
    plt.savefig(fi_path, dpi=200)
    plt.close()
    chart_paths.append(fi_path)

    plt.figure(figsize=(6, 4))
    if is_classification:
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matrice de Confusion')
        plt.xlabel('Prédiction')
        plt.ylabel('Réel')
    else:
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color='#2b5c8f')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.title('Prédictions vs Valeurs Réelles')
        plt.xlabel('Valeur Réelle')
        plt.ylabel('Valeur Prédite')

    plt.tight_layout()
    perf_path = os.path.join(CHARTS_DIR, 'model_performance.png')
    plt.savefig(perf_path, dpi=200)
    plt.close()
    chart_paths.append(perf_path)

    return {"status": "success", "chart_paths": chart_paths}
