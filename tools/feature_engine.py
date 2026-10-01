import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from tools.reliability_engine import ReliabilityEngine

class AutonomousFeatureEngine:
    def __init__(self, target_col: str, model_factory, max_sample_size: int = 5000):
        self.target_col = target_col
        self.model_factory = model_factory
        self.reliability = ReliabilityEngine(target_col)
        self.max_sample_size = max_sample_size

    def generate_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Génère des propositions de variables : ratios, logs, interactions."""
        df_feat = df.copy()
        X = df_feat.drop(columns=[self.target_col])
        num_cols = list(X.select_dtypes(include=[np.number]).columns)

        # 1. Transformations Log pour les distributions étalées (skewness > 1.0)
        for col in num_cols:
            if X[col].min() >= 0 and X[col].skew() > 1.0:
                df_feat[f"{col}_log"] = np.log1p(X[col])

        # 2. Interactions & Ratios (limités aux 10 premières colonnes)
        top_num_cols = num_cols[:10]
        for i in range(len(top_num_cols)):
            for j in range(i + 1, len(top_num_cols)):
                col1, col2 = top_num_cols[i], top_num_cols[j]
                df_feat[f"{col1}_x_{col2}"] = X[col1] * X[col2]
                if (X[col2] == 0).sum() == 0:
                    df_feat[f"{col1}_div_{col2}"] = X[col1] / (X[col2] + 1e-5)

        # 3. Encodage par fréquence des catégories
        cat_cols = list(X.select_dtypes(include=['object', 'category']).columns)
        for col in cat_cols:
            freq = X[col].value_counts(normalize=True)
            df_feat[f"{col}_freq"] = X[col].map(freq)

        return df_feat

    def run_feature_selection_loop(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """Boucle optimisée : teste uniquement un sous-ensemble restreint de candidates."""
        # 1. Échantillonnage à 3000 lignes pour accélérer la validation
        if len(df) > 3000:
            df_eval = df.sample(n=3000, random_state=42).copy()
        else:
            df_eval = df.copy()

        X_base_eval = pd.get_dummies(df_eval.drop(columns=[self.target_col]), drop_first=True)
        y_eval = df_eval[self.target_col]

        baseline_res = self.reliability.evaluate_cv_robustness(self.model_factory(), X_base_eval, y_eval, n_splits=3)
        best_score = baseline_res["mean_cv_score"]

        candidates_df_eval = self.generate_candidates(df_eval)
        
        # 2. BRIDER LE NOMBRE DE CANDIDATES (Max 8 au lieu de 60+)
        new_cols = [c for c in candidates_df_eval.columns if c not in df_eval.columns][:8]

        selected_eval = X_base_eval.copy()
        kept_features = []
        audit_log = [{
            "action": "baseline",
            "score": best_score,
            "overfitting_gap": baseline_res["overfitting_gap"]
        }]

        # 3. Boucle rapide restreinte aux 8 candidates
        for col in new_cols:
            col_data = candidates_df_eval[[col]]
            if col_data.select_dtypes(include=['object', 'category']).shape[1] > 0:
                col_data = pd.get_dummies(col_data, drop_first=True)

            X_test = pd.concat([selected_eval, col_data], axis=1)
            eval_res = self.reliability.evaluate_cv_robustness(self.model_factory(), X_test, y_eval, n_splits=3)
            new_score = eval_res["mean_cv_score"]

            gain = new_score - best_score
            if gain > 0.005 and not eval_res["is_overfitting"]:
                best_score = new_score
                selected_eval = X_test.copy()
                kept_features.append(col)
                audit_log.append({
                    "feature": col,
                    "status": "KEPT",
                    "gain": float(gain),
                    "new_score": float(best_score),
                    "overfitting_gap": eval_res["overfitting_gap"]
                })

        # Application des features retenues au dataset complet
        full_candidates = self.generate_candidates(df)
        X_full = pd.get_dummies(df.drop(columns=[self.target_col]), drop_first=True)
        for f in kept_features:
            if f in full_candidates.columns:
                col_data = full_candidates[[f]]
                if col_data.select_dtypes(include=['object', 'category']).shape[1] > 0:
                    col_data = pd.get_dummies(col_data, drop_first=True)
                X_full = pd.concat([X_full, col_data], axis=1)

        X_full[self.target_col] = df[self.target_col].values
        return X_full, audit_log