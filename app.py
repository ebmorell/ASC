import streamlit as st
import numpy as np
import joblib

# Cargar modelo
model = joblib.load("modelo_aterosclerosis_XGB.pkl")

st.title("Predicción de Aterosclerosis Subclínica en pacientes con VIH")

st.markdown("Introduce los datos clínicos del paciente:")

# Entradas del usuario
sexo = st.selectbox("Sexo", [0, 1], format_func=lambda x: "Hombre" if x == 1 else "Mujer")
edad = st.slider("Edad", 18, 90, 45)
hta = st.selectbox("Hipertensión (HTA)", [0, 1])
dm = st.selectbox("Diabetes Mellitus", [0, 1])
dlp = st.selectbox("Dislipemia", [0, 1])
imc = st.number_input("Índice de Masa Corporal (IMC)", 10.0, 50.0, 25.0)
tabaco = st.selectbox("Fumador/a", [0, 1])
alcohol = st.selectbox("Consumo de alcohol", [0, 1])
drogas_c = st.selectbox("Consumo de otras drogas", [0, 1])
via = st.selectbox("Vía de transmisión", [0, 1, 2, 3])
estadio = st.selectbox("Estadio CDC", [1, 2, 3])
sida = st.selectbox("Diagnóstico de SIDA", [0, 1])
tiempo_tar = st.number_input("Tiempo en TAR (años)", 0.0, 40.0, 5.0)
nadir = st.number_input("Nadir CD4", 0, 2000, 300)
cd4 = st.number_input("CD4 basal", 0, 2000, 500)
cd8 = st.number_input("CD8 basal", 0, 3000, 800)
col_tot = st.number_input("Colesterol total", 0.0, 500.0, 180.0)
hdl = st.number_input("HDL basal", 0.0, 150.0, 45.0)
ldl = st.number_input("LDL basal", 0.0, 300.0, 100.0)
trig = st.number_input("Triglicéridos", 0.0, 1000.0, 150.0)
ratio = st.number_input("CD4/CD8 ratio", 0.0, 5.0, 0.9)
ac_vhc = st.selectbox("Anticuerpos VHC", [0, 1])
insti = st.selectbox("Uso de INSTI", [0, 1])
itinn = st.selectbox("Uso de ITINN", [0, 1])
itian = st.selectbox("Uso de ITIAN", [0, 1])
ip_previos = st.selectbox("Uso previo de IP", [0, 1])

# Botón de predicción
if st.button("Predecir"):
    entrada = np.array([[sexo, edad, hta, dm, dlp, imc, tabaco, alcohol, drogas_c, via,
                         estadio, sida, tiempo_tar, nadir, cd4, cd8, col_tot, hdl, ldl,
                         trig, ratio, ac_vhc, insti, itinn, itian, ip_previos]])

    pred = model.predict(entrada)[0]
    proba = model.predict_proba(entrada)[0][1]

    st.success(f"Riesgo estimado: {'Aterosclerosis Subclínica' if pred == 1 else 'No Aterosclerosis'}")
    st.info(f"Probabilidad estimada: {proba:.2f}")
