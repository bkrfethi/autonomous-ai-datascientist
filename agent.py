import os
import json
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

from tools.reliability_engine import ReliabilityEngine
from tools.feature_engine import AutonomousFeatureEngine
from tools.tuning_engine import tune_hyperparameters
from catboost import CatBoostClassifier

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"))

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "audit_dataset_health",
            "description": "Vérifie les fuites de données, déséquilibres et identifiants uniques.",
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string"},
                    "target_col": {"type": "string"}
                },
                "required": ["csv_path", "target_col"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_autonomous_feature_engineering",
            "description": "Propose et valide de nouvelles features (ratios, logs, interactions) via validation croisée.",
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string"},
                    "target_col": {"type": "string"}
                },
                "required": ["csv_path", "target_col"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "optimize_model_hyperparameters",
            "description": "Exécute une recherche bayésienne Optuna sur le modèle sélectionné.",
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string"},
                    "target_col": {"type": "string"},
                    "model_name": {"type": "string"},
                    "n_trials": {"type": "integer"}
                },
                "required": ["csv_path", "target_col", "model_name"]
            }
        }
    }
]

def run_senior_autonomous_agent(csv_path: str, target_col: str, query: str) -> str:
    system_prompt = f"""Tu es un Lead AI Data Scientist de niveau Senior.
Tu disposes d'un moteur autonome complet avec détection de fuites, feature engineering et tuning Optuna.

Tu dois conduire le projet selon la boucle : PLAN -> EXPERIMENT -> EVALUATE -> CRITIC -> RE-PLAN -> REPORT.

Instructions :
1. Lance l'audit avec `audit_dataset_health`. Si un risque de leakage existe, corrige la stratégie.
2. Déclenche `run_autonomous_feature_engineering` pour tester et ne garder que les features à fort impact.
3. Optimise le meilleur modèle retenu avec `optimize_model_hyperparameters`.
4. Agis comme un Critic : Si l'overfitting gap > 10% ou le ROC-AUC < 0.80, ré-ajuste tes instructions.
5. Rédige un rapport final détaillé répondant à : "{query}"
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Fichier: {csv_path}, Cible: {target_col}"}
    ]

    for step in range(10):
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0.1
        )

        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content

        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            if func_name == "audit_dataset_health":
                df = pd.read_csv(args["csv_path"])
                engine = ReliabilityEngine(args["target_col"])
                res = engine.audit_dataset_health(df)
            elif func_name == "run_autonomous_feature_engineering":
                df = pd.read_csv(args["csv_path"])
                f_engine = AutonomousFeatureEngine(args["target_col"], lambda: CatBoostClassifier(verbose=0))
                opt_df, logs = f_engine.run_feature_selection_loop(df)
                res = {"status": "SUCCESS", "new_features_count": len(opt_df.columns) - len(df.columns), "logs": logs}
            elif func_name == "optimize_model_hyperparameters":
                df = pd.read_csv(args["csv_path"])
                res = tune_hyperparameters(df, args["target_col"], args.get("model_name", "catboost"), args.get("n_trials", 10))
            else:
                res = {"error": "Outil non reconnu"}

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(res, ensure_ascii=False)
            })

    return "Erreur : Limite d'itérations atteinte sans convergence."