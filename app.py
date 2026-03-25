import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard modèle", layout="wide")

df = pd.read_csv("dashboard_metrics.csv")

st.title("📊 Tableau de bord des performances du modèle")

st.subheader("Tableau global")
st.dataframe(df)

model = st.selectbox("Modèle", df["Model"].unique())
filtered = df[df["Model"] == model]

for metric in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]:
    fig, ax = plt.subplots(figsize=(6, 4))
    filtered.set_index("Dataset")[metric].plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title(metric)
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=0)
    st.pyplot(fig)