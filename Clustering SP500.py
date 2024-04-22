# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:19:43 2024

@author: jvazquezs
"""
#Importar librerías


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.cluster import KMeans
import statsmodels.api as sm
#______________________________________________________________________________________________________________________________
#Obtener datos


# Obtener la lista de acciones del S&P 500
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
tickers = sp500['Symbol'].tolist()

# Descargar  precios para cada acción
data = yf.download(tickers, start='2024-03-01', end='2024-04-09', interval='1wk')['Adj Close']

# Lista de tickers fallidos
failed_downloads = ['BF.B', 'BRK.B', 'GEV','SOLV']

# Crear un conjunto de tickers fallidos para facilitar la comparación
failed_set = set(failed_downloads)

# Filtrar la lista de tickers para excluir los tickers fallidos
tickers = [ticker for ticker in tickers if ticker not in failed_set]

# Descargar datos históricos de precios para cada acción de la nueva lista filtrada
data = yf.download(tickers, start='2024-03-01', end='2024-04-09', interval='1wk')['Adj Close']
#______________________________________________________________________________________________________________________________

#Retornos diarios y limpieza


# Calcular los retornos diarios
returns = data.pct_change()

# eliminamos la fila de valores NaN dada la perdida del primer registro
if returns.iloc[0].isna().any():
    # Si la primera fila contiene NaN, elimínala
    returns = returns.drop(index=returns.index[0])
    
# Obtener los retornos del índice S&P 500
sp500_returns = yf.download('^GSPC', start='2024-03-01', end='2024-04-09', interval='1wk')['Adj Close'].pct_change().dropna()
#______________________________________________________________________________________________________________________________

# Calcular las betas para cada acción


# Descargar datos históricos de precios para cada acción
data_for_betas = yf.download(tickers, start='2023-10-01', end='2024-04-09', interval='1wk')['Adj Close']

#retornos
returns_for_betas = data_for_betas.pct_change()

# Excluir la primera fila (índice 0)
returns_for_betas = returns_for_betas.iloc[1:]
    
# Obtener los retornos del índice S&P 500
sp500_returns_betas = yf.download('^GSPC', start='2023-10-01', end='2024-04-09', interval='1wk')['Adj Close'].pct_change().dropna()

# Define una función para calcular la beta
def calculate_beta(stock_returns, market_returns):
    # Añade una columna de constantes para el intercepto
    X = sm.add_constant(market_returns)
    # Realiza la regresión lineal
    model = sm.OLS(stock_returns, X)
    results = model.fit()
    # Retorna la beta (pendiente de la regresión)
    return results.params['Adj Close']

# Crear un diccionario para almacenar las betas
betas = {}

# Iterar sobre cada acción en la lista de tickers
for ticker in tickers:
    # Obtener los retornos de la acción
    stock_returns = returns_for_betas[ticker]
    
    # Asegurarse de que los retornos de la acción y del índice están alineados
    # con respecto al índice de fechas
    aligned_market_returns = sp500_returns_betas.loc[stock_returns.index]
    
    # Calcular la beta de la acción
    beta = calculate_beta(stock_returns, aligned_market_returns)
    
    # Almacenar la beta en el diccionario
    betas[ticker] = beta

#______________________________________________________________________________________________________________________________

#Estimación de rendimientos residuales


# Crear un DataFrame vacío para almacenar los rendimientos residuales
residual_returns = pd.DataFrame(index=returns.index, columns=tickers)

# Calcular los rendimientos residuales para cada acción
for ticker in tickers:
    stock_returns = returns[ticker]
    beta = betas.get(ticker, 0)  # Obtener la beta de la acción, si no está disponible, se asume como 0 #residual_returns[ticker] = stock_returns - (sp500_returns * beta)
    residual_returns[ticker] = returns[ticker] - (sp500_returns * betas[ticker])

# Rellenar los valores NaN en los rendimientos residuales con el valor anterior (forward fill)
#residual_returns.fillna(method='ffill', inplace=True)

# Calcular la suma de valores NaN en cada columna
nan_counts = residual_returns.isna().sum()
nan_columns = residual_returns.isna().all()

# Eliminar las columnas que están completamente llenas de NaN
#residual_returns = residual_returns.dropna()
#residual_returns = residual_returns.dropna(axis=1)
#_________________________________________________________________________________________________________________________________

#Matriz de correlación



# Calcular la matriz de correlación de los rendimientos residuales
correlation_matrix = residual_returns.corr()

# Mostrar la matriz de correlación
print(correlation_matrix)
correlation_matrix.describe()

# Visualizar la matriz de correlación
plt.figure(figsize=(16, 14))  # Ajusta el tamaño de la figura
sns.heatmap(
    correlation_matrix,
    annot=True,  # Muestra los valores en cada celda
    cmap='coolwarm',  # Intenta con otro colormap si es necesario
    linewidths=0.5,  # Grosor de las líneas entre celdas
    linecolor='white',  # Color de las líneas entre celdas
    annot_kws={"size": 12},  # Aumenta el tamaño de la fuente
    vmin=-1,  # Establece el valor mínimo en -1
    vmax=1   # Establece el valor máximo en 1
)
plt.title('Matriz de correlación')
plt.show()
#__________________________________________________________________________________________________________________________________

#Clusterización


#Determinar el número óptimo de clusters con elbow
inertia_values = []
num_clusters_range = range(1, 15)  # Puedes ajustar el rango de clústeres según tus necesidades

# Calcula la inercia para cada número de clústeres
for k in num_clusters_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(correlation_matrix.values)
    inertia_values.append(kmeans.inertia_)

# Grafica el método del codo
plt.figure(figsize=(8, 6))
plt.plot(num_clusters_range, inertia_values, marker='o')
plt.title('Método del codo: Inercia para diferentes números de clústeres')
plt.xlabel('Número de clústeres')
plt.ylabel('Inercia')
plt.show()

#Comprobar con silhoutte
from sklearn.metrics import silhouette_score

silhouette_scores = []
num_clusters_range = range(2, 15)  # El rango puede ajustarse según tus necesidades

# Calcula el puntaje de silueta para cada número de clústeres
for k in num_clusters_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(correlation_matrix.values)
    score = silhouette_score(correlation_matrix.values, cluster_labels)
    silhouette_scores.append(score)

# Grafica los puntajes de silueta
plt.figure(figsize=(8, 6))
plt.plot(num_clusters_range, silhouette_scores, marker='o')
plt.title('Índice de silueta para diferentes números de clústeres')
plt.xlabel('Número de clústeres')
plt.ylabel('Puntaje de silueta')
plt.show()

# Número de clusters
num_clusters = 16  # Puedes ajustar este valor según tus necesidades

# Inicializar y ajustar el modelo K-Means
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(correlation_matrix.values)

#verificar que el largo de tickers y cluster label es igual
print("Tamaño de tickers:", len(tickers))
print("Tamaño de cluster_labels:", len(cluster_labels))

# Ajustar la lista de tickers para que contenga solo los tickers presentes en residual_returns
tickers = residual_returns.columns.tolist()

# Verificar que los tamaños sean consistentes
print("Tamaño de tickers:", len(tickers))
print("Tamaño de cluster_labels:", len(cluster_labels))

# DataFrame con los símbolos y sus clústeres correspondientes
cluster_df = pd.DataFrame({
    'Symbol': tickers,
    'Cluster': cluster_labels
})

# Mostrar los primeros registros del DataFrame de clústeres
print(cluster_df.head())

# Acciones en cada clúster
print(cluster_df['Cluster'].value_counts())

# Obtener las etiquetas de cluster asignadas a cada acción
cluster_labels = kmeans.labels_

# Mostrar las etiquetas de cluster asignadas a cada acción
for i, ticker in enumerate(tickers):
    print(f'{ticker}: Cluster {cluster_labels[i]}')

# Mostrar los centroides de los clusters
print("Centroides de los clusters:")
print(kmeans.cluster_centers_)

# Ordenar los tickers y la matriz de correlación según los clústeres
cluster_df.sort_values(by='Cluster', inplace=True)
ordered_tickers = cluster_df['Symbol']
ordered_correlation_matrix = correlation_matrix.loc[ordered_tickers, ordered_tickers]

# Crear el mapa de calor
plt.figure(figsize=(16, 14))
sns.heatmap(
    ordered_correlation_matrix,
    cmap='coolwarm',
    annot=False,  # Puedes ajustar esto a True si deseas ver los valores en cada celda
    linewidths=0.5,
    linecolor='white',
    xticklabels=ordered_tickers,
    yticklabels=ordered_tickers,
    vmin=-1,
    vmax=1
)

# Título y mostrar el gráfico
plt.title('Matriz de correlación ordenada por clústeres')
plt.show()
#__________________________________________________________________________________________________________________________________

