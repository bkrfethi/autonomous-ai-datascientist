import os
import requests
import streamlit as st

st.set_page_config(page_title="Autonomous AI Data Scientist", layout="wide")

st.title("Autonomous AI Data Scientist")
st.caption("Plateforme autonome d'analyse de données, d'expérimentation ML et d'explicabilité SHAP.")

# 1. Ingestion des données
st.header("1. Ingestion des données")
uploaded_file = st.file_uploader("Chargez un fichier CSV", type=["csv"])

if uploaded_file is not None:
    # Création du dossier d'upload si inexistant
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"Fichier **{uploaded_file.name}** chargé avec succès.")

    # 2. Question métier uniquement
    st.header("2. Question métier")
    query = st.text_input(
        "Question stratégique :",
        placeholder="Ex : Quels sont les facteurs principaux influençant le risque et les leviers d'action ?"
    )

    if st.button("Lancer l'Analyse Autonome", type="primary"):
        with st.spinner("L'agent autonome inspecte le fichier, détecte la cible et exécute les expérimentations..."):
            try:
                # Appel API sans target_col (détection 100 % automatique)
                params = {
                    "file_path": file_path,
                    "query": query if query.strip() else "Analyse complète du dataset et des facteurs clés.",
                    "target_col": ""
                }
                
                response = requests.post("http://127.0.0.1:8000/ask", params=params)

                if response.status_code == 200:
                    data = response.json()
                    
                    # Extraction propre du rapport texte
                    if isinstance(data, dict):
                        report_text = data.get("report") or data.get("response") or str(data)
                    else:
                        report_text = str(data)
                    
                    st.markdown("---")
                    st.header("Rapport d'Analyse Autonome")
                    st.markdown(report_text)

                    # Affichage structuré des expérimentations si présentes
                    if isinstance(data, dict) and "experiments" in data and data["experiments"]:
                        st.markdown("---")
                        st.subheader("Historique des Expérimentations Autonomes")
                        st.json(data["experiments"])

                else:
                    st.error(f"Erreur API ({response.status_code}) : {response.text}")
                    
            except Exception as e:
                st.error(f"Erreur de connexion au serveur Uvicorn : {e}")