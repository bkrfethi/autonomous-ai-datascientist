import os
import json
from groq import Groq
from dotenv import load_dotenv

from tools.eda_engine import analyze_dataset
from tools.experiment_engine import run_experiment
from tools.agent_memory import AgentMemory
from tools.explainability import generate_shap_analysis

load_dotenv()

# Client Groq (Support de GROQ_API_KEY ou OPENAI_API_KEY)
groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
client = Groq(api_key=groq_api_key)

memory = AgentMemory()

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "analyze_dataset",
            "description": "Analyse la qualité du jeu de données (déséquilibre de classe, fuites de données, types, valeurs manquantes).",
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
            "name": "run_experiment",
            "description": "Exécute un entraînement ML. Modèles supportés : 'xgboost', 'lightgbm', 'catboost', 'logistic', 'ridge'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string"},
                    "target_col": {"type": "string"},
                    "model_name": {"type": "string"},
                    "handle_imbalance": {"type": "boolean"},
                    "log_transform_target": {"type": "boolean"}
                },
                "required": ["csv_path", "target_col", "model_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_shap_analysis",
            "description": "Génère un graphique SHAP d'explicabilité globale sous forme d'image PNG.",
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string"},
                    "target_col": {"type": "string"}
                },
                "required": ["csv_path", "target_col"]
            }
        }
    }
]

def run_autonomous_agent(csv_path: str, target_col: str, query: str) -> str:
    """Orchestre la boucle de décision autonome de l'agent AI Data Scientist."""
    memory.reset()
    
    system_prompt = f"""Tu es un Autonomous Lead AI Data Scientist.
Ton objectif est de mener des expérimentations méthodiques pour identifier le meilleur modèle ML et fournir des recommandations stratégiques.

Séquence d'exécution obligatoire :
1. Exécute `analyze_dataset` pour auditer la structure et identifier les risques (ex: déséquilibre de classe, variables à exclure).
2. Réalise au moins 2 à 3 expériences distinctes avec `run_experiment` en comparant différents algorithmes ('catboost', 'lightgbm', 'xgboost') et en ajustant 'handle_imbalance' si nécessaire.
3. Exécute `generate_shap_analysis` pour créer le visuel d'explicabilité des variables.
4. Synthétise l'ensemble des résultats dans un rapport Markdown clair :
   - Diagnostic de la donnée (EDA & Qualité)
   - Performance comparative des modèles (précision, F1-score, ROC-AUC)
   - Facteurs clés d'influence issus de l'analyse SHAP
   - Plan d'action et recommandations stratégiques pour répondre à la question : "{query}"
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Fichier: {csv_path}, Variable Cible: {target_col}. Question: {query}"}
    ]

    # Boucle d'exécution d'outils (limite max 8 itérations)
    for step in range(8):
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0.2
        )

        msg = response.choices[0].message
        messages.append(msg)

        # Si l'agent a terminé ses appels d'outils, il rédige son rapport
        if not msg.tool_calls:
            return msg.content

        # Exécution des outils demandés par le LLM
        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            if func_name == "analyze_dataset":
                res = analyze_dataset(args["csv_path"], args.get("target_col"))
            elif func_name == "run_experiment":
                res = run_experiment(
                    csv_path=args["csv_path"],
                    target_col=args["target_col"],
                    model_name=args["model_name"],
                    handle_imbalance=args.get("handle_imbalance", False),
                    log_transform_target=args.get("log_transform_target", False)
                )
                memory.log_experiment(res["experiment_config"], res["metrics"], res["top_features"])
            elif func_name == "generate_shap_analysis":
                res = generate_shap_analysis(args["csv_path"], args["target_col"])
            else:
                res = {"error": f"Outil inconnu : {func_name}"}

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(res, ensure_ascii=False)
            })

    return "Erreur : La limite d'étapes d'expérimentation a été atteinte."