import streamlit as st
import pandas as pd
import joblib

# === Load model ===
model = joblib.load("random_forest_maternal.pkl")

st.title("Maternal Health Risk Prediction App")
st.write("Masukkan data berikut untuk memprediksi risiko maternal:")

# === Input fitur dari user ===
age = st.number_input("Age", min_value=10, max_value=70, value=30)
systolic = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
diastolic = st.number_input("Diastolic BP", min_value=40, max_value=120, value=80)
bs = st.number_input("Blood Sugar (BS)", min_value=1.0, max_value=30.0, value=7.0, step=0.1)

# User input dalam Celsius
bodytemp_c = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=36.7, step=0.1)

# Convert ke Fahrenheit untuk model
bodytemp_f = bodytemp_c * 9/5 + 32

heartrate = st.number_input("Heart Rate", min_value=40, max_value=200, value=80)

# === Tombol Prediksi ===
if st.button("Predict"):
    # Buat dataframe dari input user (pakai Fahrenheit untuk model)
    sample = pd.DataFrame([{
        "Age": age,
        "SystolicBP": systolic,
        "DiastolicBP": diastolic,
        "BS": bs,
        "BodyTemp": bodytemp_f,
        "HeartRate": heartrate
    }])

    # Prediksi
    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0]

    label = "High Risk" if pred == 1 else "Low Risk"

    # Tampilkan hasil
    st.subheader("Hasil Prediksi:")
    if label == "High Risk":
        st.error(f"⚠️ {label}")
    else:
        st.success(f"✅ {label}")
