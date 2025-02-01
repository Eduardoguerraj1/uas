import streamlit as st
import joblib
import tensorflow as tf
from datetime import datetime


# Carregar o modelo treinado e o scaler
modelo = tf.keras.models.load_model('modelo_treinado.h5')
#modelo = tf.keras.models.load_model('modelo_treinado.h5')
#modelo = load_model("modelo_treinado.h5")
scaler = joblib.load("scaler.save")

# Configurar a interface do Streamlit
st.title("Previsão de Dose")
st.write("Insira os valores das variáveis para calcular a dose prevista.")

# Criar campos de entrada para as variáveis independentes
var1 = st.number_input("Distância até os Hi-Storms (aplicar 1/d^2)", value=0.0)
var2 = st.number_input("Número de Hi-Storms presentes", value=0.0)

# Entrada de data para calcular os dias desde 01/01/2021
data_entrada = st.date_input("Data atual (para cálculo de decaimento)")

# Calcular os dias desde 01/01/2021
data_base = datetime(2021, 1, 1)
dias_desde_base = (datetime.combine(data_entrada, datetime.min.time()) - data_base).days

# Mostrar o número de dias calculados (opcional)
st.write(f"Dias desde 01/01/2021: {dias_desde_base}")

var4 = st.number_input("Valor de Dose da Monitoração Direta", value=0.0)

# Botão para realizar a previsão
if st.button("Prever"):
    # Coletar os valores inseridos pelo usuário
    valores = [var1, var2, dias_desde_base, var4]

    if all(v is not None for v in valores):  # Garantir que todos os valores estão preenchidos
        # Normalizar os valores com o scaler
        valores_normalizados = scaler.transform([valores])  # Normalizar os valores de entrada

        # Fazer a previsão com o modelo treinado
        previsao_normalizada = modelo.predict(valores_normalizados)[0][0]

        # Reverter a normalização para a previsão
        min_target = scaler.data_min_[-1]  # Valor mínimo da variável dependente
        max_target = scaler.data_max_[-1]  # Valor máximo da variável dependente
        previsao_real = previsao_normalizada * (max_target - min_target) + min_target

        # Exibir o resultado
        st.write(f"O Valor Previsto de Dose Efetiva Mensal é: {previsao_real / 3:.4f} mSv/mês; "
         f"Anualizado seria: {previsao_real * 12/3:.4f} mSv/ano; "
         f"Comparar com Limite de Norma para IP (1mSv/ano) ou IOE (20mSv/ano).")
    else:
        st.warning("Por favor, insira todos os valores.")


######   streamlit run APPWEB.py
######   pyinstaller --onefile --add-data "modelo_treinado.h5;." --add-data "scaler.save;." app.py
