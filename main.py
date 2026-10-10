import os
import json
from fastapi import FastAPI, HTTPException, Query
from agent import run_senior_autonomous_agent
from tools.eda_engine import analyze_dataset

app = FastAPI(title="Autonomous AI Data Scientist API")


@app.get("/")
def read_root():
    return {"status": "API active"}


@app.post("/profile")
def profile_endpoint(file_path: str = Query(...)):
    """Profilage rapide du fichier CSV pour la sidebar Streamlit."""
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Fichier non trouvé.")
    try:
        analysis = analyze_dataset(file_path)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
def ask_agent(
    file_path: str = Query(...),
    query: str = Query(""),
    target_col: str = Query("")  # Valeur par défaut vide pour la détection automatique
):
    """Lance l'analyse autonome, la modélisation et l'explicabilité SHAP."""
    try:
        # 1. RÉINITIALISATION : Effacer la mémoire des anciennes expériences
        if os.path.exists("experiment_memory.json"):
            os.remove("experiment_memory.json")

        # 2. Exécution de l'agent
        report = run_senior_autonomous_agent(file_path, target_col, query)
        
        # 3. Récupération du nouvel historique des expériences
        experiments = []
        if os.path.exists("experiment_memory.json"):
            with open("experiment_memory.json", "r", encoding="utf-8") as f:
                experiments = json.load(f)

        return {
            "report": report,
            "experiments": experiments
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))