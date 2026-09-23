import optuna
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Silencer les logs verbeux d'Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def tune_hyperparameters(df: pd.DataFrame, target_col: str, model_name: str = "catboost", n_trials: int = 15) -> dict:
    """Optimise les hyperparamètres d'un modèle via recherche Bayésienne Optuna."""
    X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
    y = df[target_col]

    def objective(trial):
        if model_name.lower() == "catboost":
            params = {
                "iterations": trial.suggest_int("iterations", 100, 400),
                "depth": trial.suggest_int("depth", 4, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
                "verbose": 0,
                "random_seed": 42
            }
            model = CatBoostClassifier(**params)
        elif model_name.lower() == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "random_state": 42,
                "eval_metric": "logloss"
            }
            model = XGBClassifier(**params)
        else:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "random_state": 42,
                "verbose": -1
            }
            model = LGBMClassifier(**params)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_tr, y_tr)
            preds = model.predict_proba(X_va)[:, 1]
            scores.append(roc_auc_score(y_va, preds))

        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return {
        "model_name": model_name,
        "best_roc_auc": float(study.best_value),
        "best_params": study.best_params
    }