
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="Simulateur Prédictif - Transformation Digitale",
    page_icon="📊",
    layout="centered"
)

# Message d'accueil stylisé
st.markdown(
    "<h1 style='text-align: center; color: #2C3E50;'>📊 Simulateur Prédictif d'Investissement Digital</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; font-size: 18px;'>Estimez les revenus générés par un secteur en fonction du continent, des profits, dettes et nombre d’employés.</p>",
    unsafe_allow_html=True
)

# Chargement des données
df = pd.read_csv("Dataset.csv")

# Nettoyage des colonnes nécessaires
df = df.dropna(subset=['continent', 'sector', 'employees', 'revenues', 'profits', 'debt'])
df['employees'] = df['employees'].astype(float)
df['revenues'] = df['revenues'].astype(float)
df['profits'] = df['profits'].astype(float)
df['debt'] = df['debt'].astype(float)

# Sélections utilisateur
st.sidebar.header("🔧 Paramètres de simulation")
continent = st.sidebar.selectbox("🌍 Choisissez un continent", sorted(df['continent'].unique()))
sector = st.sidebar.selectbox("🏭 Choisissez un secteur", sorted(df[df['continent'] == continent]['sector'].unique()))
employees_input = st.sidebar.slider("👥 Nombre d'employés", 10, 5000, 100, step=10)
profits_input = st.sidebar.number_input("📈 Profits estimés ($)", min_value=0.0, value=100000.0, step=10000.0)
debt_input = st.sidebar.number_input("📉 Dettes estimées ($)", min_value=0.0, value=50000.0, step=10000.0)

# Filtrage pour modèle
filtered = df[(df['continent'] == continent) & (df['sector'] == sector)]

if filtered.shape[0] < 10:
    st.warning("❗ Pas assez de données pour entraîner une prédiction fiable pour cette combinaison.")
else:
    X = filtered[['employees', 'profits', 'debt']]
    y = filtered['revenues']
    model = LinearRegression()
    model.fit(X, y)

    prediction = model.predict(np.array([[employees_input, profits_input, debt_input]]))[0]
    st.success(f"💰 Revenu prévisionnel estimé : **{prediction:,.0f} $**")
