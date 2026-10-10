import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

def run_pipeline(csv_path: str, target_col: str):
    df = pd.read_csv(csv_path)
    
    if 'customer_id' in df.columns:
        df = df.drop(columns=['customer_id'])
        
    X = df.drop(columns=[target_col])
    y = df[target_col]

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()   
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    baseline = LogisticRegression()
    baseline.fit(X_train, y_train)
    y_pred_base = baseline.predict(X_test)
    y_prob_base = baseline.predict_proba(X_test)[:, 1]

    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

    print("=== BASELINE (Logistic Regression) ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_base):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred_base):.4f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob_base):.4f}\n")

    print("=== MODÈLE AVANCÉ (XGBoost) ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred_xgb):.4f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob_xgb):.4f}")

if __name__ == "__main__":
    run_pipeline('churn_data.csv', target_col='churn')