import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Configuration de la page
# =========================
st.set_page_config(page_title="Dashboard modèle", layout="wide")

# Réduire les marges latérales
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 4rem;
            padding-right: 4rem;
            max-width: 1200px;
        }
        h1, h2, h3 {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# =========================
# Chargement des données
# =========================
df = pd.read_csv("dashboard_metrics.csv")

# =========================
# Titre
# =========================
st.markdown("<h1 style='color:#4A90E2;'>📊 Tableau de bord du modèle</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Comparaison des performances des modèles Base et Balanced</p>", unsafe_allow_html=True)

# =========================
# Tableau global
# =========================
st.subheader("📋 Tableau récapitulatif")
st.dataframe(df, use_container_width=True)

# =========================
# Choix de la métrique
# =========================
metric = st.selectbox(
    "Choisir une métrique à visualiser",
    ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
)

# =========================
# Séparation des données
# =========================
base_df = df[df["Model"] == "Base"].copy()
balanced_df = df[df["Model"] == "Balanced"].copy()

# Couleurs
colors = ["#4A90E2", "#50E3C2", "#F5A623"]

# =========================
# Fonction graphique barres
# =========================
def plot_bar(data, title, metric_name):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(data["Dataset"], data[metric_name], color=colors)

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.02,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=12, fontweight="bold")
    return fig

# =========================
# Fonction graphique ligne
# =========================
def plot_line(data, title, metric_name, color):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(data["Dataset"], data[metric_name], marker="o", linewidth=2.5, color=color)

    for i, val in enumerate(data[metric_name]):
        ax.text(i, val + 0.02, f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    return fig

# =========================
# Comparaison côte à côte
# =========================
st.subheader(f"📈 Comparaison des modèles — {metric}")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Modèle Base")
    st.pyplot(plot_bar(base_df, f"{metric} - Base", metric))

with col2:
    st.markdown("### Modèle Balanced")
    st.pyplot(plot_bar(balanced_df, f"{metric} - Balanced", metric))

# =========================
# Courbes côte à côte
# =========================
st.subheader(f"📉 Évolution par dataset — {metric}")

col3, col4 = st.columns(2)

with col3:
    st.markdown("### Modèle Base")
    st.pyplot(plot_line(base_df, f"{metric} - Base", metric, "#4A90E2"))

with col4:
    st.markdown("### Modèle Balanced")
    st.pyplot(plot_line(balanced_df, f"{metric} - Balanced", metric, "#FF6F61"))

# =========================
# Comparaison directe sur un seul graphe
# =========================
st.subheader(f"🔍 Comparaison directe Base vs Balanced — {metric}")

pivot_df = df.pivot(index="Dataset", columns="Model", values=metric)

fig, ax = plt.subplots(figsize=(6, 4))
pivot_df.plot(kind="bar", ax=ax, color=["#4A90E2", "#F5A623"])

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=3)

ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title(f"{metric} - Comparaison des deux modèles", fontsize=12, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.3)
plt.xticks(rotation=0)

st.pyplot(fig)