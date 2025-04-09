import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Cargar modelo
model = joblib.load("modelo_aterosclerosis_XGB.pkl")

st.title("Predicción de Aterosclerosis Subclínica en pacientes con VIH")

st.markdown("Introduce los datos clínicos del paciente:")

# Entradas del usuario
sexo = st.selectbox("Sexo", [0, 1], format_func=lambda x: "Hombre" if x == 1 else "Mujer")
edad = st.slider("Edad", 18, 90, 45)
hta = st.selectbox("Hipertensión (HTA)", ["No", "Sí"])
hta = 1 if hta == "Sí" else 0
dm = st.selectbox("Diabetes Mellitus", ["No", "Sí"])
dm = 1 if dm == "Sí" else 0
dlp = st.selectbox("Dislipemia", ["No", "Sí"])
dlp = 1 if dlp == "Sí" else 0
imc = st.number_input("Índice de Masa Corporal (IMC)", 10.0, 50.0, 25.0)
tabaco = st.selectbox("Fumador/a", ["No", "Sí"])
tabaco = 1 if tabaco == "Sí" else 0
alcohol = st.selectbox("Consumo de alcohol", ["No", "Sí"])
alcohol = 1 if alcohol == "Sí" else 0
drogas_c = st.selectbox("Consumo de otras drogas", ["No", "Sí"])
drogas_c = 1 if drogas_c == "Sí" else 0
via_opciones = {
    "HSH": 0,
    "Drogadicción": 1,
    "Heterosexual": 2,
    "Otros/desconocido": 3
}

via_texto = st.selectbox("Vía de transmisión", list(via_opciones.keys()))
via = via_opciones[via_texto]

estadio = st.selectbox("Estadio CDC", [1, 2, 3])
sida = st.selectbox("Diagnóstico de SIDA", ["No", "Sí"])
sida = 1 if sida == "Sí" else 0

tiempo_tar = st.number_input("Tiempo en TAR (años)", 0.0, 40.0, 5.0)
nadir = st.number_input("Nadir CD4", 0, 2000, 300)
cd4 = st.number_input("CD4 basal", 0, 2000, 500)
cd8 = st.number_input("CD8 basal", 0, 3000, 800)

col_tot = st.number_input("Colesterol total", 0.0, 500.0, 180.0)
hdl = st.number_input("HDL basal", 0.0, 150.0, 45.0)
ldl = st.number_input("LDL basal", 0.0, 300.0, 100.0)
trig = st.number_input("Triglicéridos", 0.0, 1000.0, 150.0)

ratio = st.number_input("CD4/CD8 ratio", 0.0, 5.0, 0.9)

ac_vhc = st.selectbox("Anticuerpos VHC", ["No", "Sí"])
ac_vhc = 1 if ac_vhc == "Sí" else 0

insti = st.selectbox("Uso de INSTI", ["No", "Sí"])
insti = 1 if insti == "Sí" else 0

itinn = st.selectbox("Uso de ITINN", ["No", "Sí"])
itinn = 1 if itinn == "Sí" else 0

itian = st.selectbox("Uso de ITIAN", ["No", "Sí"])
itian = 1 if itian == "Sí" else 0

ip_previos = st.selectbox("Uso previo de IP", ["No", "Sí"])
ip_previos = 1 if ip_previos == "Sí" else 0


# Botón de predicción
if st.button("Predecir"):
    entrada = np.array([[sexo, edad, hta, dm, dlp, imc, tabaco, alcohol, drogas_c, via,
                         estadio, sida, tiempo_tar, nadir, cd4, cd8, col_tot, hdl, ldl,
                         trig, ratio, ac_vhc, insti, itinn, itian, ip_previos]])

    pred = model.predict(entrada)[0]
    proba = model.predict_proba(entrada)[0][1]
    porcentaje = proba * 100

    st.success(f"Resultado estimado: {'Aterosclerosis Subclínica' if pred == 1 else 'No Aterosclerosis'}")
    st.info(f"Probabilidad estimada de ASC: {porcentaje:.1f}%")

    # Visualización del riesgo
    st.markdown("### Visualización del riesgo estimado")

    fig, ax = plt.subplots(figsize=(6, 1.2))

    # Colores por zonas
    ax.axhspan(0, 1, xmin=0, xmax=0.33, color='lightblue', label="Baja")
    ax.axhspan(0, 1, xmin=0.33, xmax=0.66, color='gold', label="Media")
    ax.axhspan(0, 1, xmin=0.66, xmax=1, color='salmon', label="Alta")

    # Indicador de riesgo del paciente
    ax.plot([proba], [0.5], marker='o', color='black', markersize=12)

    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.33, 0.66, 1])
    ax.set_xticklabels(['0%', '33%', '66%', '100%'])
    ax.set_title("Nivel de riesgo de ASC")
    plt.tight_layout()

    st.pyplot(fig)
