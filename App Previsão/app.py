import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import pickle
from datetime import datetime, timedelta
from calendar import monthrange

# Carrega modelo
model = pickle.load(open('model_lgbm.pkl', 'rb'))

# Carrega dados
df_agrupado = pd.read_csv('df_agrupado.csv')
vendas_completa = pd.read_csv('vendas_completa.csv')
vendas_completa['DATA_ATEND'] = pd.to_datetime(vendas_completa['DATA_ATEND'])

st.title("📈 Previsão de Vendas por SKU e Filial")

# Define opções de filial
filiais_disponiveis = sorted(df_agrupado['COD_FILIAL'].unique())
filiais_opcoes = [str(f) for f in filiais_disponiveis if f in [101032, 101042]]
filiais_opcoes.append("Ambas (101032 + 101042)")

filial_selecionada = st.selectbox("Selecione a Filial", filiais_opcoes)
skus = sorted(df_agrupado['SKU'].unique())
sku = st.selectbox("Selecione o SKU", skus)
ano = st.number_input("Ano", min_value=2018, max_value=2030, value=2024)
mes = st.number_input("Mês", min_value=1, max_value=12, value=3)
dia = st.number_input("Dia", min_value=1, max_value=31, value=28)
dias_vizualizacao = st.number_input("Quantidade de dias para visualizar no gráfico (depois da data escolhida)", min_value=1, max_value=365, value=60)

# Corrige datas inválidas
try:
    data_usuario = datetime(int(ano), int(mes), int(dia))
except ValueError:
    dia = monthrange(int(ano), int(mes))[1]
    data_usuario = datetime(int(ano), int(mes), int(dia))

dia_semana = data_usuario.weekday()
fim_de_semana = 1 if dia_semana >= 5 else 0
inicio_mes = 1 if dia <= 5 else 0
ultimo_dia_mes = monthrange(int(ano), int(mes))[1]
fim_mes = 1 if dia >= ultimo_dia_mes - 4 else 0

# Define quais filiais filtrar
if filial_selecionada == "Ambas (101032 + 101042)":
    filiais_filtro = [101032, 101042]
else:
    filiais_filtro = [int(filial_selecionada)]

filial_previsao = filiais_filtro[0]

linha_base = df_agrupado[
    (df_agrupado['COD_FILIAL'] == filial_previsao) &
    (df_agrupado['SKU'] == sku) &
    (df_agrupado['MES'] == mes) &
    (df_agrupado['DIA'] <= dia)
].sort_values(by='DIA', ascending=False).head(1)

if linha_base.empty:
    linha_base = df_agrupado[
        (df_agrupado['COD_FILIAL'] == filial_previsao) &
        (df_agrupado['SKU'] == sku) &
        (df_agrupado['MES'] == mes)
    ].sort_values(by='DIA').head(1)

if linha_base.empty:
    st.warning("⚠️ Nenhuma venda.")
else:
    dados = linha_base.iloc[0]
    entrada = np.array([[filial_previsao, sku, dados['CATEGORIA'], ano, mes, dados['SUM_FATUR'], dia,
                         dia_semana, dados['MM3'], dados['MM7'], dados['MM14'], dados['STD7'],
                         dados['TENDENCIA_CURTA'], fim_de_semana, inicio_mes, fim_mes]])
    pred = model.predict(entrada)
    st.success(f"Previsão de vendas (Filial {filial_previsao}): {pred[0]:.2f} unidades")

# Filtra vendas para filiais selecionadas 
vendas_filtradas = vendas_completa[
    (vendas_completa['SKU'] == sku) &
    (vendas_completa['COD_FILIAL'].isin(filiais_filtro))
]

if not vendas_filtradas.empty:
    data_limite_inicial = data_usuario + timedelta(days=1)
    data_limite_final = data_usuario + timedelta(days=int(dias_vizualizacao))

    vendas_periodo = vendas_filtradas[
        (vendas_filtradas['DATA_ATEND'] >= data_limite_inicial) &
        (vendas_filtradas['DATA_ATEND'] <= data_limite_final)
    ]

    if not vendas_periodo.empty:
        vendas_por_dia = vendas_periodo.groupby('DATA_ATEND')['QTD_VENDA'].sum().reset_index()
        titulo_filial = filial_selecionada
        fig = px.line(vendas_por_dia,
                      x='DATA_ATEND',
                      y='QTD_VENDA',
                      title=f"Histórico de Vendas para os próximos {dias_vizualizacao} dias após {data_usuario.strftime('%Y-%m-%d')} | SKU {sku} | Filial {titulo_filial}",
                      labels={'DATA_ATEND': 'Data', 'QTD_VENDA': 'Quantidade Vendida (Soma)'})
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig)
    else:
        st.warning("⚠️ Não há vendas registradas nesse intervalo.")
else:
    st.warning("⚠️ Nenhuma venda registrada para esse SKU e filial(s).")
