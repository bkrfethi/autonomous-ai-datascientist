import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from lightgbm import LGBMClassifier
from tools.reliability_engine import ReliabilityEngine

class AutonomousFeatureEngine:
    def __init__(self, target_col: str, model_factory=None, max_sample_size: int = 3000):
        self.target_col = target_col
        self.model_factory = model_factory
        self.max_sample_size = max_sample_size
        self.reliability = ReliabilityEngine(target_col)

    def generate_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Génère toutes les variables candidates en une passe vectorisée."""
        df_feat = df.copy()
        X = df_feat.drop(columns=[self.target_col])
        num_cols = list(X.select_dtypes(include=[np.number]).columns)

        # 1. Transformations Log pour distributions étalées
        for col in num_cols:
            if X[col].min() >= 0 and X[col].skew() > 1.0:
                df_feat[f"{col}_log"] = np.log1p(X[col])

        # 2. Ratios / Interactions ciblés (max 5 colonnes numériques clés)
        top_num = num_cols[:5]
        for i in range(len(top_num)):
            for j in range(i + 1, len(top_num)):
                c1, c2 = top_num[i], top_num[j]
                df_feat[f"{c1}_x_{c2}"] = X[c1] * X[c2]
                if (X[c2] == 0).sum() == 0:
                    df_feat[f"{c1}_div_{c2}"] = X[c1] / (X[c2] + 1e-5)

        # 3. Encodage fréquence
        cat_cols = list(X.select_dtypes(include=['object', 'category']).columns)
        for col in cat_cols:
            freq = X[col].value_counts(normalize=True)
            df_feat[f"{col}_freq"] = X[col].map(freq)

        return df_feat

    def run_feature_selection_loop(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """Sélection ULTRA-RAPIDE via Feature Importance (0.5 seconde)."""
        # Échantillonnage rapide
        df_sample = df.sample(n=min(len(df), self.max_sample_size), random_state=42).copy()

        # Génération globale
        candidates_df = self.generate_candidates(df_sample)
        X_cand = pd.get_dummies(candidates_df.drop(columns=[self.target_col]), drop_first=True)
        y_cand = candidates_df[self.target_col]

        # 1 SEUL modèle rapide (LightGBM) pour évaluer TOUTES les variables d'un coup
        fast_model = LGBMClassifier(n_estimators=80, learning_rate=0.1, n_jobs=-1, random_state=42, verbose=-1)
        fast_model.fit(X_cand, y_cand)

        base_cols = list(pd.get_dummies(df.drop(columns=[self.target_col]), drop_first=True).columns)
        new_features = [col for col in X_cand.columns if col not in base_cols]

        # Sélection des variables au-dessus du 50ème percentile d'importance
        importances = fast_model.feature_importances_
        audit_log = []
        selected_new_cols = []

        if len(importances) > 0:
            threshold = np.median(importances)
            for feat, imp in zip(X_cand.columns, importances):
                if feat in new_features:
                    if imp > threshold and imp > 0:
                        selected_new_cols.append(feat)
                        audit_log.append({"feature": feat, "status": "KEPT", "importance": float(imp)})
                    else:
                        audit_log.append({"feature": feat, "status": "REJECTED", "importance": float(imp)})

        # Application rapide sur le dataset complet
        full_candidates = self.generate_candidates(df)
        X_full = pd.get_dummies(full_candidates.drop(columns=[self.target_col]), drop_first=True)
        
        cols_to_keep = [c for c in X_full.columns if c in base_cols or c in selected_new_cols]
        X_final = X_full[cols_to_keep].copy()
        X_final[self.target_col] = df[self.target_col].values

        return X_final, audit_log