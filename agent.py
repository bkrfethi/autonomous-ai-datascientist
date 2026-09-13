import os
import json
from groq import Groq
from dotenv import load_dotenv
from tools.eda_engine import analyze_dataset
from tools.experiment_engine import run_experiment
from tools.agent_memory import AgentMemory
from tools.explainability import generate_shap_analysis
# Charge les variables d'environnement
load_dotenv()

# ✅ INITIALISATION CORRECTE DU CLIENT GROQ
client = Groq(
    api_key=os.getenv("OPENAI_API_KEY")
)

memory = AgentMemory()

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "analyze_dataset",
            "description": "Analyse la qualité du jeu de données (déséquilibre, fuites de données, types, valeurs manquantes).",
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
            "description": "Exécute une expérience ML. Modèles supportés : 'xgboost', 'lightgbm', 'catboost', 'logistic', 'ridge'.",
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
        "description": "Génère un graphique SHAP d'explicabilité de modèle sous forme d'image PNG.",
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
    memory.reset()
    
    system_prompt = f"""Tu es un Autonomous Lead AI Data Scientist.
Ton objectif est de mener des expérimentations méthodiques pour trouver le MEILLEUR modèle ML.

Processus obligatoire :
1. Démarre par `analyze_dataset` pour auditer le dataset et repérer les anomalies (déséquilibre, skewness, fuites).
2. Conduis au moins 2 à 3 expériences distinctes avec `run_experiment` en variant les algorithmes ('catboost', 'lightgbm', 'xgboost') ou en activant 'handle_imbalance' / 'log_transform_target' si le diagnostic l'exige.
3. Compare les métriques obtenues et sélectionne la meilleure expérience.
4. Rédige un rapport final très structuré avec :
   - Diagnostic EDA & Qualité de la donnée
   - Tableau comparatif des expériences menées
   - Interprétabilité du meilleur modèle (variables les plus contributives)
   - Recommandations stratégiques métiers pour répondre à la question : "{query}"
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Fichier: {csv_path}, Variable Cible: {target_col}. Question: {query}"}
    ]

    # Boucle d'exécution de l'agent (maximum 8 itérations d'outils)
    for _ in range(8):
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0.2
        )

        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            # L'agent a fini d'expérimenter et rédige son rapport final
            return msg.content

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
                # Sauvegarde automatique dans la mémoire
                memory.log_experiment(res["experiment_config"], res["metrics"], res["top_features"])

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(res, ensure_ascii=False)
            })

    return "Erreur : Limite d'expérimentations atteinte."
# import os
# import json
# from openai import OpenAI
# from dotenv import load_dotenv
# from tools.data_profiler import profile_dataset
# from tools.ml_trainer import train_and_evaluate

# load_dotenv()

# # Configuration spécifique pour Groq
# client = OpenAI(
#     api_key=os.getenv("OPENAI_API_KEY"),
#     base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
# )

# # 1. Définition des outils pour le LLM (Function Calling Schema)
# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "profile_dataset",
#             "description": "Inspecte et analyse la structure d'un fichier CSV (colonnes, types, valeurs manquantes, statistiques de base).",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "csv_path": {
#                         "type": "string", 
#                         "description": "Chemin du fichier CSV à analyser."
#                     }
#                 },
#                 "required": ["csv_path"]
#             }
#         }
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "train_and_evaluate",
#             "description": "Entraîne des modèles ML sur un jeu de données et renvoie les métriques ainsi que l'importance des variables.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "csv_path": {
#                         "type": "string", 
#                         "description": "Chemin du fichier CSV."
#                     },
#                     "target_col": {
#                         "type": "string", 
#                         "description": "Nom de la colonne cible (target) à prédire."
#                     }
#                 },
#                 "required": ["csv_path", "target_col"]
#             }
#         }
#     },
#     {
#     "type": "function",
#     "function": {
#         "name": "generate_charts",
#         "description": "Génère des visuels analytiques (importance des variables et performances) sous forme d'images PNG.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "csv_path": {"type": "string", "description": "Chemin du fichier CSV."},
#                 "target_col": {"type": "string", "description": "Nom de la colonne cible."}
#             },
#             "required": ["csv_path", "target_col"]
#         }
#     }
# }
# ]

# def run_agent(user_query: str, csv_path: str) -> str:
#     """Orchestre la prise de décision du LLM et l'exécution des outils."""
    
#     messages = [
#         {
#             "role": "system",
#             "content": (
#                 "Tu es un Autonomous AI Data Scientist d'élite. "
#                 "Ton rôle est de répondre à la question de l'utilisateur en exécutant les outils nécessaires. "
#                 "1. Commence toujours par profiler le jeu de données pour identifier la structure et la colonne cible. "
#                 "2. Entraîne ensuite les modèles sur la colonne cible. "
#                 "3. Entraîne ensuite les modèles et génère les graphiques d'analyse avec generate_charts. "
#                 "Utilise un ton professionnel mais accessible. Structure ta réponse en 3 parties : "
#                 "- Résumé exécutif (1 paragraphe)"
#                 "- Analyse des résultats (métriques, feature importance)"
#                 "- Recommandations stratégiques (3 bullet points maximum)"
#             )
#         },
#         {
#             "role": "user",
#             "content": f"Fichier disponible : {csv_path}\nQuestion de l'utilisateur : {user_query}"
#         }
#     ]

#     print("🚀 Lancement de l'agent...")
#     step_count = 0
    
#     # Boucle d'exécution des outils
#     while True:
#         step_count += 1
#         print(f"\n🔄 Étape {step_count} - Appel au LLM...")
        
#         try:
#             response = client.chat.completions.create(
#                 model="openai/gpt-oss-120b" , # OUI, il est réactivé !,  # Modèle Groq (le plus performant)
#                 messages=messages,
#                 tools=tools,
#                 tool_choice="auto",
#                 temperature=0.3  # Plus précis pour les tâches structurées
#             )
#         except Exception as e:
#             print(f"❌ Erreur API : {e}")
#             return f"Erreur lors de l'appel à l'API : {str(e)}"
        
#         response_message = response.choices[0].message
#         messages.append(response_message)

#         # Si le LLM ne demande plus d'outils, il a rédigé sa réponse finale
#         if not response_message.tool_calls:
#             print("✅ Agent a terminé son analyse")
#             return response_message.content

#         # Exécution des outils demandés par le LLM
#         for tool_call in response_message.tool_calls:
#             func_name = tool_call.function.name
#             args = json.loads(tool_call.function.arguments)

#             print(f"🤖 Exécution : {func_name} avec {args}")

#             try:
#                 if func_name == "profile_dataset":
#                     result = profile_dataset(args["csv_path"])
#                 elif func_name == "train_and_evaluate":
#                     result = train_and_evaluate(args["csv_path"], args["target_col"])
#                 elif func_name == "generate_charts":
#                     result = generate_charts(args["csv_path"], args["target_col"])
#                 else:
#                     result = {"error": f"Outil inconnu : {func_name}"}
#             except Exception as e:
#                 result = {"error": f"Erreur lors de l'exécution de {func_name} : {str(e)}"}
#                 print(f"⚠️ {result['error']}")

#             messages.append({
#                 "role": "tool",
#                 "tool_call_id": tool_call.id,
#                 "content": json.dumps(result, ensure_ascii=False)
#             })

# if __name__ == "__main__":
#     # Test avec des données d'exemple
#     query = "Quels clients risquent de partir et quels sont les facteurs principaux de ce churn ?"
    
#     print("="*60)
#     print("🧠 AUTONOMOUS AI DATA SCIENTIST - GROQ VERSION")
#     print("="*60)
    
#     # Assure-toi que le fichier existe !
#     csv_file = "uploads/churn_data.csv"
#     if not os.path.exists(csv_file):
#         print(f"⚠️ Fichier {csv_file} non trouvé. Création d'un fichier de test...")
#         # Crée un fichier de test
#         import pandas as pd
#         import numpy as np
#         np.random.seed(42)
#         n = 500
#         data = {
#             'customer_id': range(1, n+1),
#             'age': np.random.randint(18, 70, n),
#             'monthly_spending': np.random.uniform(10, 200, n),
#             'login_frequency': np.random.poisson(5, n),
#             'support_tickets': np.random.poisson(1.5, n),
#             'churn': np.random.choice([0, 1], n, p=[0.7, 0.3])
#         }
#         df = pd.DataFrame(data)
#         os.makedirs('uploads', exist_ok=True)
#         df.to_csv(csv_file, index=False)
#         print(f"✅ Fichier de test créé : {csv_file}")
    
#     report = run_agent(query, csv_file)
    
#     print("\n" + "="*60)
#     print("📊 RAPPORT FINAL")
#     print("="*60)
#     print(report)
#     print("\n" + "="*60)