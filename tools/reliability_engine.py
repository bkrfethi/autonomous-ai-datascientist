import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def compute_dataset_hash(df: pd.DataFrame) -> str:
    """Génère un hash SHA-256 unique pour garantir la traçabilité des données."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

class ReliabilityEngine:
    def __init__(self, target_col: str, threshold_leakage: float = 0.90):
        self.target_col = target_col
        self.threshold_leakage = threshold_leakage

    def audit_dataset_health(self, df: pd.DataFrame) -> dict:
        """Détecte les fuites de données, les colonnes constantes et le déséquilibre."""
        issues = []
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        # 1. Détection des ID / Colonnes à trop forte cardinalité
        for col in X.columns:
            is_string_or_obj = pd.api.types.is_string_dtype(X[col]) or pd.api.types.is_object_dtype(X[col])
            if X[col].nunique() == len(X) and is_string_or_obj:
                issues.append(f"LEAKAGE_RISK: La colonne '{col}' est un identifiant unique.")

        # 2. Corrélation directe suspecte avec la cible
        num_cols = X.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            corr = abs(y.corr(X[col]))
            if corr >= self.threshold_leakage:
                issues.append(f"CRITICAL_LEAKAGE: '{col}' a une corrélation de {corr:.2f} avec la cible.")

        # 3. Ratio de déséquilibre
        class_counts = y.value_counts(normalize=True)
        imbalance_ratio = class_counts.min() / class_counts.max()

        return {
            "dataset_hash": compute_dataset_hash(df),
            "is_healthy": len(issues) == 0,
            "issues": issues,
            "imbalance_ratio": float(imbalance_ratio)
        }

    def evaluate_cv_robustness(self, model, X: pd.DataFrame, y: pd.Series, n_splits: int = 3) -> dict:
        """Évalue la robustesse en CV Stratifiée et calcule l'écart Train / Validation."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        train_scores, val_scores = [], []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_tr, y_tr)

            if hasattr(model, "predict_proba"):
                tr_preds = model.predict_proba(X_tr)[:, 1]
                va_preds = model.predict_proba(X_va)[:, 1]
            else:
                tr_preds = model.predict(X_tr)
                va_preds = model.predict(X_va)

            train_scores.append(roc_auc_score(y_tr, tr_preds))
            val_scores.append(roc_auc_score(y_va, va_preds))

        mean_tr = float(np.mean(train_scores))
        mean_va = float(np.mean(val_scores))
        gap = mean_tr - mean_va

        return {
            "mean_cv_score": mean_va,
            "train_cv_score": mean_tr,
            "overfitting_gap": gap,
            "is_overfitting": gap > 0.15
        }