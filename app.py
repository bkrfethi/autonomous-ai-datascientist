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
                st.markdown("### 📋 Rapport généré par l'AI Agent")
                st.markdown(data["report"])
                
                # Affichage dynamique des graphiques si générés
                if os.path.exists("charts/feature_importance.png") and os.path.exists("charts/model_performance.png"):
                    st.divider()
                    st.markdown("### 📊 Visualisations du Modèle")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image("charts/feature_importance.png", caption="Importance des Variables", use_container_width=True)
                    with col2:
                        st.image("charts/model_performance.png", caption="Performance du Modèle", use_container_width=True)
            else:
                st.error(f"Erreur API ({response.status_code}) : {response.text}")
else:
    st.info("Veuillez charger un fichier CSV dans le panneau latéral pour commencer.")