import logging
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse
import plotly.express as px
from plotly.io import to_html
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
sf_salaries_path = os.path.join(settings.BASE_DIR, "app", "Salaries.csv")

# Create your views here.

def home_page(request):
    logging.info(f"Pagina de home aberta")
    try:
        graficos = analisar_data_set()
        logging.info(f"Graficos gerados: " + str(len(graficos)))
        if graficos is not None:
            return render(request, 'home.html', {'graficos_html' : graficos})
    except Exception as e:
        return HttpResponse("Ocorreu um erro inesperado a analisar os dados - Mensagem de Erro: ", e , status=500)

# ========================================================
# Funcaoes de dataset

def analisar_data_set():
    logging.info(" --- Start do processo de análise do dataset --- ")
    try:
        df = transformar_em_dataframe()
        graficos_html = gerar_graficos(df)
        logging.info(" --- Fim do processo de análise do dataset --- ")
        return graficos_html
    except Exception as e:
        logging.error(f"Erro no processo de análise do dataset: {e}")
        return []

def transformar_em_dataframe():
    logging.info(" --- Start do processo de transformar o dataset em DataFrame pandas ---")
    try:
        if not os.path.isfile(sf_salaries_path):
            logging.error(f"Arquivo não encontrado: {sf_salaries_path}")
            return pd.DataFrame()
        df = pd.read_csv(sf_salaries_path, low_memory=False)
        print(df.columns)
        df_sem_col_id = df.drop(df.columns[0], axis=1)
        logging.info(" --- Fim do processo de transformar o dataset em DataFrame pandas ---")
        return df_sem_col_id
    except Exception as e:
        logging.error(f"Erro no processo de transformar o dataset em DataFrame pandas: {e}")
        return pd.DataFrame()

def learning(df):
    try:
        logging.info("pre processamento de dados, jobTitle para numeros")
        label_encoder = LabelEncoder()
        df['JobTitle_encoded'] = label_encoder.fit_transform(df['JobTitle'])
        # vars importantes para previsao
        X = df[['JobTitle_encoded', 'BasePay', 'OvertimePay', 'OtherPay', 'Benefits']]
        y = df['TotalPay']  # objetivo
        logging.info(f"dividindo dados em treino e teste")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info("treinando o random forest")
        modelo = RandomForestRegressor(n_estimators=100, random_state=42)
        modelo.fit(X_train, y_train)
        return modelo
    except Exception as e:
        logging.error(f"Erro no machine learning de previsao de salario por funcao")

# ======================================================
# funcoes de grafico

def gerar_graficos(df):
    logging.info(" --- Start do processo de gerar graficos html --- ")
    try:
        graficos_html = []
        graficos_html = gerar_grafico_tempo_por_salario_medio(df, graficos_html)
        logging.info(" --- Fim do processo de gerar graficos html --- ")
        return graficos_html
    except Exception as e:
        logging.error(f"Erro ao tentar gerar gráficos html: {e}")
        raise  # Levantar novamente a exceção para depuração


def gerar_grafico_tempo_por_salario_medio(df, graficos_html):
    #converter as colunas year e totalpay para formato numerico e deletar espacos vazios
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['TotalPay'] = pd.to_numeric(df['TotalPay'], errors='coerce')

    #calculo da media de salario anual
    df_media = df.groupby('Year')['TotalPay'].mean().reset_index()


    df_media['Year_sin'] = np.sin(2 * np.pi * (df_media['Year'] - df_media['Year'].min()) / 12)
    df_media['Year_cos'] = np.cos(2 * np.pi * (df_media['Year'] - df_media['Year'].min()) / 12)

    #regressao polinomial configuracao
    X = df_media[['Year_sin', 'Year_cos']].values
    y = df_media['TotalPay'].values
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly.fit_transform(X)

    modelo = LinearRegression()
    modelo.fit(X_poly, y)

    #previsao iterativa dos anos de 2014 ate 2023
    previsoes = []
    for ano in range(2014, 2023):
        year_sin = np.sin(2 * np.pi * (ano - df_media['Year'].min()) / 12)
        year_cos = np.cos(2 * np.pi * (ano - df_media['Year'].min()) / 12)
        X_atual = [[year_sin, year_cos]]
        X_atual_poly = poly.transform(X_atual)
        previsao_salario = modelo.predict(X_atual_poly)[0]
        previsoes.append({'Year': ano, 'PredictedTotalPay': previsao_salario})

    #criacao do dataframe para previsoes
    df_previsao = pd.DataFrame(previsoes)

    #juntando os dados historicos com as previsoes
    df_final = pd.concat([df_media, df_previsao], ignore_index=True, sort=False)

    #geracao do grafico
    grafico = px.line(df_final, x='Year', y='TotalPay', title="Evolução da Média de Salários com Previsão Futura")
    grafico.update_traces(line=dict(color='black', width=6, dash='solid'))

    #adicionando coloracao e espessura na linha do grafico
    grafico.add_scatter(
        x=df_previsao['Year'],
        y=df_previsao['PredictedTotalPay'],
        mode='lines+markers',  # Linha e pontos
        name='Previsão',
        line=dict(color='red', width=4, dash='dash'),  # Linha vermelha e mais grossa
        marker=dict(color='red', size=8, symbol='circle')  # Pontos vermelhos e maiores
    )
    #legenda
    grafico.update_layout(
        legend_title="Salário Médio",
        legend=dict(x=0.8, y=0.9)
    )
    graficos_html.append(to_html(grafico, full_html=False))
    return graficos_html


