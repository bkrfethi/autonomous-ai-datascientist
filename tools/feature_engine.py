import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from tools.reliability_engine import ReliabilityEngine

class AutonomousFeatureEngine:
    def __init__(self, target_col: str, model_factory):
        self.target_col = target_col
        self.model_factory = model_factory
        self.reliability = ReliabilityEngine(target_col)

    def generate_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Génère des propositions de variables : ratios, logs, interactions."""
        df_feat = df.copy()
        X = df_feat.drop(columns=[self.target_col])
        num_cols = list(X.select_dtypes(include=[np.number]).columns)

        # 1. Transformations Log pour les distributions étalées (skewness > 1.0)
        for col in num_cols:
            if X[col].min() >= 0 and X[col].skew() > 1.0:
                df_feat[f"{col}_log"] = np.log1p(X[col])

        # 2. Interactions & Ratios entre colonnes numériques
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                col1, col2 = num_cols[i], num_cols[j]
                # Produit
                df_feat[f"{col1}_x_{col2}"] = X[col1] * X[col2]
                # Ratio sécurisé contre division par zéro
                if (X[col2] == 0).sum() == 0:
                    df_feat[f"{col1}_div_{col2}"] = X[col1] / (X[col2] + 1e-5)

        # 3. Encodage par fréquence des catégories
        cat_cols = list(X.select_dtypes(include=['object', 'category']).columns)
        for col in cat_cols:
            freq = X[col].value_counts(normalize=True)
            df_feat[f"{col}_freq"] = X[col].map(freq)

        return df_feat

    def run_feature_selection_loop(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """Boucle expérimentale EDA -> Proposal -> Validation -> Keep/Reject."""
        # Baseline
        X_base = pd.get_dummies(df.drop(columns=[self.target_col]), drop_first=True)
        y = df[self.target_col]
        
        baseline_res = self.reliability.evaluate_cv_robustness(self.model_factory(), X_base, y)
        best_score = baseline_res["mean_cv_score"]
        
        candidates_df = self.generate_candidates(df)
        new_cols = [c for c in candidates_df.columns if c not in df.columns]

        selected_df = X_base.copy()
        audit_log = [{
            "action": "baseline",
            "score": best_score,
            "overfitting_gap": baseline_res["overfitting_gap"]
        }]

        # Tester chaque feature candidate individuellement
        for col in new_cols:
            col_data = candidates_df[[col]]
            if col_data.select_dtypes(include=['object', 'category']).shape[1] > 0:
                col_data = pd.get_dummies(col_data, drop_first=True)

            X_test = pd.concat([selected_df, col_data], axis=1)
            
            # Évaluation CV
            eval_res = self.reliability.evaluate_cv_robustness(self.model_factory(), X_test, y)
            new_score = eval_res["mean_cv_score"]
            
            # Condition de conservation : Gain de performance sans surapprentissage excessif
            gain = new_score - best_score
            if gain > 0.005 and not eval_res["is_overfitting"]:
                best_score = new_score
                selected_df = X_test.copy()
                audit_log.append({
                    "feature": col,
                    "status": "KEPT",
                    "gain": float(gain),
                    "new_score": float(best_score),
                    "overfitting_gap": eval_res["overfitting_gap"]
                })
            else:
                audit_log.append({
                    "feature": col,
                    "status": "REJECTED",
                    "score_diff": float(gain),
                    "is_overfitting": eval_res["is_overfitting"]
                })

        selected_df[self.target_col] = y
        return selected_df, audit_log