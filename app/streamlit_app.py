import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(layout="wide", page_title="Fraud Detection Dashboard")

st.title("Fraud Detection Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("data/raw/creditcard.csv")
    return df

@st.cache_resource
def load_model():
    model = joblib.load("models/fraud_xgb_model.pkl")
    return model

df = load_data()
model = load_model()

# Sidebar
st.sidebar.header("📊 Navigation")
view = st.sidebar.radio("Aller vers :", ["Vue d'ensemble", "Faire une prédiction"])

if view == "Vue d'ensemble":
    st.subheader("Aperçu des données")
    st.write(df.head())

    st.subheader("Distribution des classes")
    fig = px.histogram(df, x="Class", title="Classe (0 = normal, 1 = fraude)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Valeurs statistiques")
    st.write(df.describe())

elif view == "Faire une prédiction":
    st.subheader("🔍 Prédiction d'une transaction")
    uploaded_file = st.file_uploader("Charger un fichier CSV", type="csv")

    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
        st.write("Aperçu du fichier :", input_df.head())

        prediction = model.predict(input_df)
        prediction_prob = model.predict_proba(input_df)[:, 1]

        input_df['Prediction'] = prediction
        input_df['Probabilité de fraude'] = prediction_prob

        st.subheader("Résultat de la prédiction")
        st.write(input_df[['Prediction', 'Probabilité de fraude']])

        fraud_detected = input_df['Prediction'].sum()
        st.success(f"🚨 {fraud_detected} transaction(s) suspecte(s) détectée(s)")