import pytest
import pandas as pd
import numpy as np
from tools.reliability_engine import ReliabilityEngine
from tools.tuning_engine import tune_hyperparameters

@pytest.fixture
def mock_dataset():
    np.random.seed(42)
    df = pd.DataFrame({
        "customer_id": [f"ID_{i}" for i in range(100)],
        "age": np.random.randint(18, 70, size=100),
        "monthly_spending": np.random.uniform(20, 500, size=100),
        "churn": np.random.choice([0, 1], size=100, p=[0.8, 0.2])
    })
    return df

def test_data_leakage_detection(mock_dataset):
    engine = ReliabilityEngine(target_col="churn")
    audit = engine.audit_dataset_health(mock_dataset)
    
    assert not audit["is_healthy"]
    assert any("customer_id" in issue for issue in audit["issues"])

def test_optuna_tuning_execution(mock_dataset):
    clean_df = mock_dataset.drop(columns=["customer_id"])
    result = tune_hyperparameters(clean_df, target_col="churn", model_name="catboost", n_trials=2)
    
    assert "best_roc_auc" in result
    assert "best_params" in result
    assert result["best_roc_auc"] > 0.0