import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tools.data_profiler import profile_dataset
from tools.ml_trainer import train_and_evaluate
from agent import run_agent
from agent import run_autonomous_agent
app = FastAPI(title="Autonomous AI Data Scientist API")

# Configuration CORS pour autoriser le futur frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"status": "online", "message": "Autonomous AI Data Scientist API"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Reçoit un fichier CSV et le sauvegarde sur le serveur."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Le fichier doit être au format CSV.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {"message": "Fichier téléversé avec succès", "file_path": file_path}

@app.post("/profile")
def profile_data(file_path: str):
    """Exécute l'analyse exploratoire du dataset."""
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Fichier non trouvé.")
    return profile_dataset(file_path)

@app.post("/train")
def train_model(file_path: str, target_col: str):
    """Lance l'entraînement des modèles ML."""
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Fichier non trouvé.")
    try:
        results = train_and_evaluate(file_path, target_col)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
def ask_agent(file_path: str, query: str, target_col: str = "churn"):
    try:
        report = run_autonomous_agent(csv_path=file_path, target_col=target_col, query=query)
        
        # Charger l'historique de la mémoire
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