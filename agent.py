import os
import re
import json
import numpy as np
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

from tools.reliability_engine import ReliabilityEngine
from tools.feature_engine import AutonomousFeatureEngine
from tools.tuning_engine import tune_hyperparameters
from lightgbm import LGBMClassifier

load_dotenv()

# Initialisation du client Groq
groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
client = Groq(api_key=groq_api_key)

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


def load_and_preprocess_dataset(csv_path: str, target_col: str):
    """Charge le dataset, nettoie la casse, supprime les ID uniques et assainit les noms de colonnes."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier non trouvé : {csv_path}")

    df = pd.read_csv(csv_path)

    # 1. Sensibilité à la casse pour la cible
    matching_cols = [col for col in df.columns if col.lower() == target_col.lower()]
    if matching_cols:
        actual_target_col = matching_cols[0]
    else:
        raise ValueError(f"La colonne '{target_col}' est introuvable. Colonnes présentes : {list(df.columns)}")

    # 2. Suppression automatique des colonnes d'identifiants uniques (ex: customerID)
    cols_to_drop = []
    for col in df.columns:
        if col != actual_target_col and (df[col].dtype == 'object' or isinstance(df[col].iloc[0], str)):
            if df[col].nunique() >= 0.8 * len(df):
                cols_to_drop.append(col)
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # 3. Encodage Binaire / Numérique de la cible (Yes/No -> 1/0)
    if df[actual_target_col].dtype == 'object' or isinstance(df[actual_target_col].iloc[0], str):
        val_map = {'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0}
        df[actual_target_col] = df[actual_target_col].astype(str).str.strip().str.lower().map(val_map)
        if df[actual_target_col].isna().any():
            df[actual_target_col] = pd.factorize(df[actual_target_col])[0]

    # 4. Conversion des colonnes numériques masquées (ex: TotalCharges avec " ")
    for col in df.columns:
        if col != actual_target_col and df[col].dtype == 'object':
            converted = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')
            if converted.notna().sum() > 0.5 * len(df):
                df[col] = converted.fillna(converted.median())

    # 5. Nettoyage des noms de colonnes pour LightGBM (remplacement des espaces et parenthèses)
    clean_cols = [re.sub(r'[^\w]', '_', col) for col in df.columns]
    df.columns = clean_cols
    actual_target_col = re.sub(r'[^\w]', '_', actual_target_col)

    return df, actual_target_col


def default_converter(o):
    """Sérialiseur JSON pour éviter les erreurs sur les types NumPy."""
    if isinstance(o, (np.integer, np.int64)):
        return int(o)
    elif isinstance(o, (np.floating, np.float64)):
        return float(o)
    elif isinstance(o, np.ndarray):
        return o.tolist()
    elif isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"Objet de type {type(o)} non sérialisable")


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

            try:
                if func_name == "audit_dataset_health":
                    df, actual_target = load_and_preprocess_dataset(args["csv_path"], args["target_col"])
                    engine = ReliabilityEngine(actual_target)
                    res = engine.audit_dataset_health(df)

                elif func_name == "run_autonomous_feature_engineering":
                    df, actual_target = load_and_preprocess_dataset(args["csv_path"], args["target_col"])
                    f_engine = AutonomousFeatureEngine(actual_target, lambda: LGBMClassifier(verbose=-1, n_jobs=-1))
                    opt_df, logs = f_engine.run_feature_selection_loop(df)
                    res = {
                        "status": "SUCCESS",
                        "new_features_count": len(opt_df.columns) - len(df.columns),
                        "logs": logs
                    }

                elif func_name == "optimize_model_hyperparameters":
                    df, actual_target = load_and_preprocess_dataset(args["csv_path"], args["target_col"])
                    res = tune_hyperparameters(
                        df,
                        actual_target,
                        args.get("model_name", "lightgbm"),
                        args.get("n_trials", 3)
                    )
                else:
                    res = {"error": "Outil non reconnu"}

            except Exception as e:
                res = {"error": str(e)}

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(res, default=default_converter, ensure_ascii=False)
            })

    return "Erreur : Limite d'itérations atteinte sans convergence."


# Alias de compatibilité pour main.py
run_autonomous_agent = run_senior_autonomous_agent