import mlflow
import pandas as pd
from typing import Dict, Any
from catboost import CatBoostClassifier
from tools.reliability_engine import ReliabilityEngine
from tools.feature_engine import AutonomousFeatureEngine

class MultiAgentOrchestrator:
    def __init__(self, target_col: str):
        self.target_col = target_col
        self.reliability = ReliabilityEngine(target_col)

    def model_factory(self):
        return CatBoostClassifier(verbose=0, random_state=42)

    def run_pipeline(self, df_path: str) -> Dict[str, Any]:
        mlflow.set_experiment("autonomous_ds_experiment")
        
        with mlflow.start_run():
            df = pd.read_csv(df_path)
            mlflow.log_param("dataset_path", df_path)
            
            # 1. DATA ANALYST AGENT — Audit de santé
            health = self.reliability.audit_dataset_health(df)
            mlflow.log_param("dataset_hash", health["dataset_hash"])
            mlflow.log_param("is_healthy", health["is_healthy"])

            if not health["is_healthy"]:
                # Nettoyage automatique des fuites
                for issue in health["issues"]:
                    if "CRITICAL_LEAKAGE" in issue or "LEAKAGE_RISK" in issue:
                        col_to_drop = issue.split("'")[1]
                        df = df.drop(columns=[col_to_drop])

            # 2. EXPERIEMENTER & FEATURE ENGINE AGENT
            feature_engine = AutonomousFeatureEngine(self.target_col, self.model_factory)
            optimized_df, feature_logs = feature_engine.run_feature_selection_loop(df)

            # 3. EVALUATOR AGENT — Robustesse finale
            X = optimized_df.drop(columns=[self.target_col])
            y = optimized_df[self.target_col]
            final_eval = self.reliability.evaluate_cv_robustness(self.model_factory(), X, y)

            # 4. CRITIC AGENT — Décision et Re-planification
            critic_rejected = False
            feedback = "Modèle valide pour déploiement."
            
            if final_eval["is_overfitting"]:
                critic_rejected = True
                feedback = "CRITIC REJECT: Gap Train/Val > 10%. Régularisation nécessaire."
            elif final_eval["mean_cv_score"] < 0.70:
                critic_rejected = True
                feedback = "CRITIC REJECT: ROC-AUC insuffisant (< 0.70)."

            # Logging MLflow
            mlflow.log_metrics({
                "cv_score": final_eval["mean_cv_score"],
                "std_score": final_eval["std_cv_score"],
                "overfitting_gap": final_eval["overfitting_gap"]
            })
            mlflow.log_dict(feature_logs, "feature_engineering_log.json")

            return {
                "health_check": health,
                "feature_logs": feature_logs,
                "metrics": final_eval,
                "critic_decision": {
                    "rejected": critic_rejected,
                    "feedback": feedback
                }
            }