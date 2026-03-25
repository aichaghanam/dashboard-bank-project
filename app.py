import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(page_title="Dashboard modèle", layout="centered")

# Charger les données
df = pd.read_csv("dashboard_metrics.csv")

# Titre
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>📊 Tableau de bord du modèle</h1>", unsafe_allow_html=True)

# Centrage du contenu
col1, col2, col3 = st.columns([1, 3, 1])

with col2:

    st.subheader("📋 Données")
    st.dataframe(df)

    # Sélection du modèle
    model = st.selectbox("Choisir le modèle", df["Model"].unique())
    filtered = df[df["Model"] == model]

    # Palette de couleurs
    colors = ["#4A90E2", "#50E3C2", "#F5A623"]

    metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]

    for metric in metrics:

        st.markdown(f"### 📈 {metric}")

        # ===== GRAPH BARRES =====
        fig, ax = plt.subplots(figsize=(5, 3))

        bars = ax.bar(filtered["Dataset"], filtered[metric], color=colors)

        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                    f"{height:.2f}",
                    ha='center', va='bottom')

        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title(metric)

        st.pyplot(fig)

        # ===== GRAPH LIGNE =====
        fig2, ax2 = plt.subplots(figsize=(5, 3))

        ax2.plot(filtered["Dataset"], filtered[metric],
                 marker='o', linewidth=2, color="#FF6F61")

        for i, val in enumerate(filtered[metric]):
            ax2.text(i, val, f"{val:.2f}", ha='center')

        ax2.set_ylim(0, 1)
        ax2.set_title(f"{metric} (évolution)")
        ax2.set_ylabel("Score")

        st.pyplot(fig2)

        # ===== GRAPH CAMEMBERT =====
        fig3, ax3 = plt.subplots(figsize=(4, 4))

        ax3.pie(filtered[metric],
                labels=filtered["Dataset"],
                autopct="%1.2f%%",
                colors=colors,
                startangle=90)

        ax3.set_title(f"Répartition {metric}")

        st.pyplot(fig3)

        st.markdown("---")