import streamlit as st
import pandas as pd
import joblib
import google.generativeai as genai
import os
from dotenv import load_dotenv

# === Load env dan model ===
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = joblib.load("random_forest_maternal.pkl")
llm_model = genai.GenerativeModel("gemini-2.0-flash")

st.title("Maternal Health Risk Prediction App")

# === Input user (2 kolom) ===
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10, max_value=70, value=30)
    systolic = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
    diastolic = st.number_input("Diastolic BP", min_value=40, max_value=120, value=80)

with col2:
    bs = st.number_input("Blood Sugar (BS)", min_value=1.0, max_value=30.0, value=7.0, step=0.1)
    bodytemp_c = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=36.7, step=0.1)
    heartrate = st.number_input("Heart Rate", min_value=40, max_value=200, value=80)

# Convert suhu ke Fahrenheit (untuk model)
bodytemp_f = bodytemp_c * 9/5 + 32

# --- Prediksi ---
if st.button("Predict"):
    sample = pd.DataFrame([{
        "Age": age,
        "SystolicBP": systolic,
        "DiastolicBP": diastolic,
        "BS": bs,
        "BodyTemp": bodytemp_f,
        "HeartRate": heartrate
    }])
    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0]
    label = "High Risk" if pred == 1 else "Low Risk"

    # Simpan hasil prediksi ke session_state
    st.session_state["last_label"] = label
    st.session_state["last_input"] = {
        "age": age,
        "systolic": systolic,
        "diastolic": diastolic,
        "bs": bs,
        "bodytemp_c": bodytemp_c,
        "heartrate": heartrate,
    }

# --- Selalu tampilkan hasil prediksi terakhir jika ada ---
if "last_label" in st.session_state:
    st.subheader("Hasil Prediksi (terakhir):")
    if st.session_state["last_label"] == "High Risk":
        st.error(f"⚠️ {st.session_state['last_label']}")
    else:
        st.success(f"✅ {st.session_state['last_label']}")

    # --- Rekomendasi ---
    if st.button("Rekomendasi"):
        with st.spinner("🔄 Sedang menyiapkan rekomendasi..."):
            user_input = st.session_state["last_input"]
            label = st.session_state["last_label"]

            prompt = f"""
            Berikan satu rekomendasi singkat (1 paragraf) untuk ibu dengan kondisi:
            Age: {user_input['age']}, Systolic BP: {user_input['systolic']}, 
            Diastolic BP: {user_input['diastolic']}, Blood Sugar: {user_input['bs']}, 
            Body Temperature: {user_input['bodytemp_c']:.1f} °C, Heart Rate: {user_input['heartrate']}.
            Risiko yang terdeteksi: {label}.
            Saran harus singkat, jelas, dan praktis.
            """
            try:
                response = llm_model.generate_content(prompt)
                st.info(response.text)
            except Exception as e:
                st.error("❌ Gagal mendapatkan rekomendasi dari Gemini.")
                st.caption(str(e))
