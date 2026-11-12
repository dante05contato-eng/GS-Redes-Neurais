import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

st.set_page_config(page_title="Classificador de Ocorrências - SP", layout="centered")

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('best_mlp_model.keras')
    preprocessor = joblib.load('preprocessor.pkl')
    return model, preprocessor

model, preprocessor = load_assets()

st.title("Classificador de Ocorrências: Roubo vs Furto")
st.caption("Segurança Pública Inteligente em São Paulo | Modelo MLP")

st.markdown("""
Esta ferramenta utiliza uma rede neural do tipo **Multilayer Perceptron (MLP)** para avaliar as características de um boletim registrado e identificar a compatibilidade com a tipificação penal.
""")

with st.form("form_ocorrencia"):
    col1, col2 = st.columns(2)
    
    with col1:
        regiao = st.selectbox("Região de SP", ['Centro', 'Zona Leste', 'Zona Sul', 'Zona Norte', 'Zona Oeste'])
        periodo = st.selectbox("Período do Dia", ['Madrugada', 'Manhã', 'Tarde', 'Noite'])
        
    with col2:
        dia_semana = st.selectbox("Dia da Semana", ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo'])
        tipo_local = st.selectbox("Tipo de Local", ['Via pública', 'Comércio/Serviços', 'Residência', 'Transporte coletivo'])
        
    submit = st.form_submit_button("Classificar Ocorrência")

if submit:
    entrada = pd.DataFrame([{
        'REGIAO': regiao,
        'PERIODO': periodo,
        'DIA_SEMANA': dia_semana,
        'TIPO_LOCAL': tipo_local
    }])
    
    X_input = preprocessor.transform(entrada)
    prob_roubo = float(model.predict(X_input)[0][0])
    prob_furto = 1.0 - prob_roubo
    
    classe_final = "Roubo" if prob_roubo >= 0.5 else "Furto"
    
    st.divider()
    st.subheader(f"Classificação Estimada: **{classe_final.upper()}**")
    
    col_r, col_f = st.columns(2)
    col_r.metric("Probabilidade de Roubo", f"{prob_roubo * 100:.1f}%")
    col_f.metric("Probabilidade de Furto", f"{prob_furto * 100:.1f}%")
    
    st.progress(prob_roubo)
    
    st.info("""
    **Interpretação técnica:** Considerando que uma ocorrência aconteceu com essas características de tempo e espaço, o modelo atribui as probabilidades acima para a tipificação penal. **Este modelo não prevê o risco nem a probabilidade de um crime acontecer no local.**
    """)