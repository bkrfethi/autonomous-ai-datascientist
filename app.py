import os
import requests
import pandas as pd
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Autonomous AI Data Scientist",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Autonomous AI Data Scientist")
st.caption("Plateforme autonome d'analyse de données, d'expérimentation ML et d'explicabilité SHAP.")

# Barre latérale : Ingestion des données
st.sidebar.header("1. Ingestion des données")
uploaded_file = st.sidebar.file_uploader("Chargez un fichier CSV", type=["csv"])

if uploaded_file:
    # Sauvegarde du fichier dans le dossier uploads
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success(f"Fichier `{uploaded_file.name}` prêt.")

    # Profilage rapide dans la sidebar
    if st.sidebar.button("📊 Profiler le dataset"):
        with st.spinner("Analyse rapide de la structure..."):
            res = requests.post(f"{API_URL}/profile", params={"file_path": file_path})
            if res.status_code == 200:
                profile = res.json()
                st.sidebar.metric("Lignes", profile.get("num_rows", 0))
                st.sidebar.metric("Colonnes", profile.get("num_columns", 0))
                st.sidebar.json(profile.get("columns", []))
            else:
                st.sidebar.error("Erreur lors du profilage du fichier.")

    st.divider()

    # Zone principale : Configuration & Question Métier
    st.subheader("2. Paramètres & Question métier")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        target_col = st.text_input("Variable cible (Target Column) :", value="churn")
    with col2:
        user_query = st.text_input(
            "Question stratégique :",
            value="Quels clients risquent de partir et quels sont les facteurs principaux de ce churn ?"
        )

    if st.button("🚀 Lancer l'analyse autonome", type="primary"):
        with st.spinner("L'agent analyse la donnée, mène des expériences, génère le graphique SHAP et rédige le rapport..."):
            response = requests.post(
                f"{API_URL}/ask",
                params={"file_path": file_path, "query": user_query, "target_col": target_col}
            )

            if response.status_code == 200:
                data = response.json()
                
                # 1. Rapport Markdown
                st.markdown("### 📋 Rapport d'Analyse Autonome")
                st.markdown(data.get("report", "Aucun rapport généré."))
                
                # 2. Tableau comparatif des expériences ML
                if data.get("experiments"):
                    st.divider()
                    st.markdown("### 🔬 Historique des Expérimentations Autonomes")
                    exp_df = []
                    for exp in data["experiments"]:
                        row = {
                            "Exp ID": exp["experiment_id"],
                            "Modèle": exp["config"]["model_name"].upper(),
                            "Imbalance handlé": exp["config"]["handle_imbalance"]
                        }
                        row.update(exp["metrics"])
                        exp_df.append(row)
                    st.dataframe(pd.DataFrame(exp_df), use_container_width=True)

                # 3. Graphique SHAP
                if os.path.exists("charts/shap_summary.png"):
                    st.divider()
                    st.markdown("### 🔍 Explicabilité SHAP du Modèle")
                    st.image(
                        "charts/shap_summary.png",
                        caption="Impact et importance des caractéristiques (SHAP Beeswarm Plot)",
                        use_container_width=True
                    )
            else:
                st.error(f"Erreur API ({response.status_code}) : {response.text}")
else:
    st.info("Veuillez charger un fichier CSV dans le panneau latéral pour commencer.")