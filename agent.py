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
                    "target_col": {"type": "string", "description": "Laissez vide ou 'auto' pour détection automatique."}
                },
                "required": ["csv_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_autonomous_feature_engineering",
            "description": "Propose et valide de nouvelles features via sélection par importance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string"},
                    "target_col": {"type": "string"}
                },
                "required": ["csv_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "optimize_model_hyperparameters",
            "description": "Exécute une recherche bayésienne Optuna.",
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string"},
                    "target_col": {"type": "string"},
                    "model_name": {"type": "string"},
                    "n_trials": {"type": "integer"}
                },
                "required": ["csv_path"]
            }
        }
    }
]


def detect_target_column(df: pd.DataFrame) -> str:
    """Détecte automatiquement la colonne cible la plus probable."""
    cols = list(df.columns)
    
    # 1. Noms fréquents de cibles
    common_targets = ['target', 'churn', 'survived', 'label', 'class', 'exited', 'outcome', 'response', 'y', 'status', 'deposit']
    for col in cols:
        if col.lower().strip() in common_targets:
            return col

    # 2. Dernière colonne si elle a peu de valeurs uniques (ex: classification)
    last_col = cols[-1]
    if df[last_col].nunique() < (len(df) * 0.5):
        return last_col

    # 3. Recherche de la dernière colonne binaire
    for col in reversed(cols):
        if df[col].nunique() == 2:
            return col

    return last_col


def load_and_preprocess_dataset(csv_path: str, target_col: str = None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier non trouvé : {csv_path}")

    df = pd.read_csv(csv_path)

    # Détection automatique si la cible n'est pas fournie ou est vide
    if not target_col or str(target_col).strip() in ["", "None", "auto"]:
        actual_target_col = detect_target_column(df)
    else:
        matching_cols = [col for col in df.columns if col.lower() == target_col.lower()]
        if matching_cols:
            actual_target_col = matching_cols[0]
        else:
            actual_target_col = detect_target_column(df)

    # Nettoyage des identifiants uniques (ex: IDs)
    cols_to_drop = []
    for col in df.columns:
        if col != actual_target_col and (df[col].dtype == 'object' or isinstance(df[col].iloc[0], str)):
            if df[col].nunique() >= 0.8 * len(df):
                cols_to_drop.append(col)
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Conversion de la cible en Binaire (0/1) si textuelle
    if df[actual_target_col].dtype == 'object' or isinstance(df[actual_target_col].iloc[0], str):
        val_map = {'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0}
        df[actual_target_col] = df[actual_target_col].astype(str).str.strip().str.lower().map(val_map)
        if df[actual_target_col].isna().any():
            df[actual_target_col] = pd.factorize(df[actual_target_col])[0]

    # Conversion des colonnes numériques masquées
    for col in df.columns:
        if col != actual_target_col and df[col].dtype == 'object':
            converted = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')
            if converted.notna().sum() > 0.5 * len(df):
                df[col] = converted.fillna(converted.median())

    # Assainissement des noms de colonnes
    clean_cols = [re.sub(r'[^\w]', '_', col) for col in df.columns]
    df.columns = clean_cols
    actual_target_col = re.sub(r'[^\w]', '_', actual_target_col)

    return df, actual_target_col


def default_converter(o):
    if isinstance(o, (np.integer, np.int64)):
        return int(o)
    elif isinstance(o, (np.floating, np.float64)):
        return float(o)
    elif isinstance(o, np.ndarray):
        return o.tolist()
    elif isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"Objet de type {type(o)} non sérialisable")


def run_senior_autonomous_agent(csv_path: str, target_col: str = "", query: str = "") -> str:
    system_prompt = f"""Tu es un Lead AI Data Scientist de niveau Senior.
Tu exécutes la chaîne d'analyse de manière 100% autonome.

CONSIGNE STRICTE :
- Ne pose JAMAIS de question à l'utilisateur.
- Lance DIRECTEMENT l'outil `audit_dataset_health` au premier tour.
- Si la colonne cible n'est pas fournie, passe `target_col=""` : le système la détectera automatiquement.

Étapes à enchaîner sans interruption :
1. `audit_dataset_health`
2. `run_autonomous_feature_engineering`
3. `optimize_model_hyperparameters`
4. Rédaction du rapport complet en markdown répondant à : "{query}"
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Analyse le fichier '{csv_path}'. Cible transmise : '{target_col}'."}
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
                provided_target = args.get("target_col", target_col)
                df, actual_target = load_and_preprocess_dataset(args["csv_path"], provided_target)

                if func_name == "audit_dataset_health":
                    engine = ReliabilityEngine(actual_target)
                    res = engine.audit_dataset_health(df)
                    res["detected_target"] = actual_target

                elif func_name == "run_autonomous_feature_engineering":
                    f_engine = AutonomousFeatureEngine(actual_target, lambda: LGBMClassifier(verbose=-1, n_jobs=-1))
                    opt_df, logs = f_engine.run_feature_selection_loop(df)
                    res = {
                        "status": "SUCCESS",
                        "target_used": actual_target,
                        "new_features_count": len(opt_df.columns) - len(df.columns),
                        "logs": logs
                    }

                elif func_name == "optimize_model_hyperparameters":
                    res = tune_hyperparameters(
                        df,
                        actual_target,
                        args.get("model_name", "lightgbm"),
                        args.get("n_trials", 3)
                    )
                    res["target_used"] = actual_target
                else:
                    res = {"error": "Outil non reconnu"}

            except Exception as e:
                res = {"error": str(e)}

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(res, default=default_converter, ensure_ascii=False)
            })

    return "Erreur : Limite d'itérations atteinte."


run_autonomous_agent = run_senior_autonomous_agent