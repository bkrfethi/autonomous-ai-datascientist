import optuna
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

def tune_hyperparameters(df: pd.DataFrame, target_col: str, model_name: str = "lightgbm", n_trials: int = 3, max_sample_size: int = 3000) -> dict:
    """Tuning bayésien ultra-rapide avec LightGBM multi-threadé."""
    df_tune = df.sample(n=min(len(df), max_sample_size), random_state=42)

    X = pd.get_dummies(df_tune.drop(columns=[target_col]), drop_first=True)
    y = df_tune[target_col]

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.25, random_state=42)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2),
            "num_leaves": trial.suggest_int("num_leaves", 15, 31),
            "n_jobs": -1,
            "random_state": 42,
            "verbose": -1
        }
        model = LGBMClassifier(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_va)[:, 1]
        return roc_auc_score(y_va, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return {
        "model_name": "lightgbm",
        "best_roc_auc": float(study.best_value),
        "best_params": study.best_params
    }