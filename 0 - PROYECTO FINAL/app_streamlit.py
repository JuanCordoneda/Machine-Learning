import os
import joblib
import streamlit as st
import numpy as np

# --- CONFIGURACIÓN DE LA APP ---
st.set_page_config(page_title="PROFIT TUNE", layout="centered", page_icon="🎵")
st.title("🎧 PROFIT TUNE")
st.header("Predicción de Éxito Musical")
st.markdown(
    "Simulador interactivo basado en tu modelo de **Machine Learning** entrenado con atributos de audio del género electrónico."
)

# --- CARGA DE MODELOS ---
rf_path = "modelo_random_forest.pkl"
xgb_path = "modelo_xgboost.pkl"
scaler_path = "scaler.pkl"

rf_model = joblib.load(rf_path)
xgb_model = joblib.load(xgb_path)
scaler = joblib.load(scaler_path)


if not os.path.exists(rf_path):
    st.error(f"❌ No se encontró el modelo en: {rf_path}")
else:
    modelo = joblib.load(rf_path)
    scaler = joblib.load(scaler_path)
    
    # === ENTRADAS DEL USUARIO (variables finales del modelo) ===
    st.subheader("🎛️ Ingresá las características de la canción:")
    anio = st.number_input("Año de lanzamiento", 2000, 2025, 2017)
    tempo = st.slider("Tempo (BPM)", 60, 200, int(103.019))
    duracion = st.number_input("Duración (ms)", 10000, 500000, 247160)
    compas = st.slider("Beats (beats por compás)", 3, 7, 4)
    volumen = st.slider("Volumen (dB)", -60.0, 0.0, -6.769)
    energia = st.slider("Energía", 0.0, 1.0, 0.635)
    tonalidad = st.slider("Tonalidad (0–11)", 0, 11, 11)
    habla = st.slider("Vocales", 0.0, 1.0, 0.0317)
    acustica = st.slider("Acústica", 0.0, 1.0, 0.0498)
    valencia = st.slider("Valencia (Positividad)", 0.0, 1.0, 0.446)
    modo = st.selectbox("Modo musical", [0, 1], index=0, format_func=lambda x: "Menor" if x == 0 else "Mayor")

    # === FEATURES DERIVADAS (se calculan igual que en el dataset original) ===
    energia_val = energia * valencia
    dance_vol = abs(volumen) * 1   # antes era Bailabilidad * abs(volumen) pero bailabilidad se eliminó
    tempo_norm = tempo / duracion
    acustica_inv = 1 - acustica

    # === VECTOR FINAL QUE USA EL MODELO ===
    X = np.array([[
        anio,
        duracion,
        compas,
        energia,
        tonalidad,
        volumen,
        modo,
        habla,
        acustica,
        valencia,
        tempo,
        energia_val,
        dance_vol,
        tempo_norm,
        acustica_inv
    ]])

    if st.button("🔮 Predecir Éxito con Ambos Modelos"):
        X_scaled = scaler.transform(X)

        # --- RANDOM FOREST ---
        pred_rf = rf_model.predict(X_scaled)[0]
        prob_rf = rf_model.predict_proba(X_scaled)[0][1]

        # --- XGBOOST ---
        pred_xgb = xgb_model.predict(X_scaled)[0]
        prob_xgb = xgb_model.predict_proba(X_scaled)[0][1]

        # === CONCLUSIÓN GLOBAL ===
        st.markdown("### 🧠 Conclusión general del sistema")
        st.info(
            "Ambos modelos procesaron tus características de audio. "
            "Random Forest tiende a ser **más estable y conservador**, "
            "mientras que XGBoost suele capturar **patrones más complejos** y arriesgados. "
            "Compará ambas predicciones para tomar una decisión informada."
        )

        st.markdown("---")

        # === RANDOM FOREST ===
        st.subheader("🌳 Random Forest")
        if pred_rf == 1:
            st.success(f"ÉXITO PROBABLE — Confianza: {prob_rf*100:.2f}%")
        else:
            st.error(f"BAJA probabilidad — {prob_rf*100:.2f}%")

        st.markdown("---")

        # === XGBOOST ===
        st.subheader("⚡ XGBoost")
        if pred_xgb == 1:
            st.success(f"ÉXITO PROBABLE — Confianza: {prob_xgb*100:.2f}%")
        else:
            st.error(f"BAJA probabilidad — {prob_xgb*100:.2f}%")

        # === CIERRE ===
        st.markdown("---")
        st.caption(
            "El sistema combina dos enfoques distintos para ofrecerte una visión más completa del potencial comercial del track."
        )


