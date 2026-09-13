import streamlit as st
import os
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Autonomous AI Data Scientist",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Autonomous AI Data Scientist")
st.caption("Plateforme autonome d'analyse de données, de modélisation ML et de recommandation stratégique.")

# Barre latérale : Chargement des données
st.sidebar.header("1. Ingestion des données")
uploaded_file = st.sidebar.file_uploader("Chargez un fichier CSV", type=["csv"])

if uploaded_file:
    # Sauvegarde temporaire du fichier
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success(f"Fichier `{uploaded_file.name}` prêt.")

    # Profilage rapide dans la sidebar
    if st.sidebar.button("📊 Profiler le dataset"):
        with st.spinner("Analyse du jeu de données en cours..."):
            res = requests.post(f"{API_URL}/profile", params={"file_path": file_path})
            if res.status_code == 200:
                profile = res.json()
                st.sidebar.metric("Lignes", profile["num_rows"])
                st.sidebar.metric("Colonnes", profile["num_columns"])
                st.sidebar.json(profile["columns"])

    st.divider()

    # Zone principale : Question & Agent
    st.subheader("2. Posez votre question métier")
    user_query = st.text_input(
        "Exemple de question :",
        value="Quels clients risquent de partir et quels sont les facteurs principaux de ce churn ?"
    )

    if st.button(" Lancer l'analyse autonome", type="primary"):
        with st.spinner("L'agent AI analyse les données, entraîne les modèles et rédige le rapport..."):
            response = requests.post(
                f"{API_URL}/ask",
                params={"file_path": file_path, "query": user_query}
            )

            if response.status_code == 200:
                data = response.json()
                
                # Rapport Markdown
                st.markdown("### 📋 Rapport d'Analyse Autonome")
                st.markdown(data["report"])
                
                # Affichage du tableau d'expérimentations
                if data.get("experiments"):
                    st.divider()
                    st.markdown("### 🔬 Historique des Expérimentations Autonomes")
                    exp_df = []
                    for exp in data["experiments"]:
                        row = {"Exp ID": exp["experiment_id"], "Modèle": exp["config"]["model_name"].upper()}
                        row.update(exp["metrics"])
                        exp_df.append(row)
                    st.dataframe(pd.DataFrame(exp_df), use_container_width=True)

                # Affichage du graphique SHAP
                if os.path.exists("charts/shap_summary.png"):
                    st.divider()
                    st.markdown("### 🔍 Explicabilité SHAP du Modèle")
                    st.image("charts/shap_summary.png", caption="Analyse SHAP des contributions de variables", use_container_width=True)
else:
    st.info("Veuillez charger un fichier CSV dans le panneau latéral pour commencer.")