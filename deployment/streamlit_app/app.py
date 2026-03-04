import streamlit as st
import requests

st.title("Heart Attack Prediction (Real-time)")

# User input
AgeCategory = st.selectbox("Age Category", ["18-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80+"])
Sex = st.selectbox("Sex", ["Male","Female"])
HadAngina = st.selectbox("Had Angina", ["Yes","No"])
HadCOPD = st.selectbox("Had COPD", ["Yes","No"])
HadDiabetes = st.selectbox("Had Diabetes", ["Yes","No","Yes, but only during pregnancy"])
PhysicalActivities = st.selectbox("Physical Activities", ["Yes","No"])
SmokerStatus = st.selectbox("Smoker Status", ["Never smoked","Former smoker","Current smoker - some days","Current smoker - every day"])
AlcoholDrinkers = st.selectbox("Alcohol Drinkers", ["Yes","No"])

data = {
    "AgeCategory": AgeCategory,
    "Sex": Sex,
    "HadAngina": HadAngina,
    "HadCOPD": HadCOPD,
    "HadDiabetes": HadDiabetes,
    "PhysicalActivities": PhysicalActivities,
    "SmokerStatus": SmokerStatus,
    "AlcoholDrinkers": AlcoholDrinkers
}

if st.button("Predict"):
    url = "https://<YOUR_CLOUD_RUN_URL>/predict"  # Replace with Cloud Run URL
    response = requests.post(url, json=data)
    if response.status_code == 200:
        result = response.json()
        st.success(f"Prediction: {result['predictions']}")
    else:
        st.error(response.text)
