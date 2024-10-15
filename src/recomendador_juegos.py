#Importamos librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import resample

# Cargar los datos
games_df = pd.read_csv('output_steam_games.csv')
reviews_df = pd.read_csv('reseñas.csv')

# Preprocesamiento de games_df
games_df['Géneros_Juego'] = games_df['Géneros_Juego'].fillna('')
games_df['Etiquetas_Juego'] = games_df['Etiquetas_Juego'].fillna('')
games_df['contenido'] = games_df['Géneros_Juego'] + ' ' + games_df['Etiquetas_Juego']
games_df['contenido'] = games_df['contenido'].str.lower()

# Preprocesamiento de reviews_df
# Revisamos que todos los datos de la columna 'Analisis_sentimientos' sean numéricos y no estén vacíos
reviews_df['Analisis_sentimientos'] = pd.to_numeric(reviews_df['Analisis_sentimientos'], errors='coerce')

# Eliminamos filas vacías con valores nulos en 'Analisis_sentimientos'
reviews_df = reviews_df.dropna(subset=['Analisis_sentimientos'])

# Verificamos los valores únicos en 'Analisis_sentimientos'
print("Valores únicos en 'Analisis_sentimientos':", reviews_df['Analisis_sentimientos'].unique())

# Balanceo de datos

# Verificamos la distribución actual de los sentimientos
print("\nDistribución de sentimientos antes del balanceo:")
print(reviews_df['Analisis_sentimientos'].value_counts())

# Hacemos un gráfico para visualizar la distribución de los sentimientos antes del balanceo
plt.figure(figsize=(6, 4))
sns.countplot(x='Analisis_sentimientos', data=reviews_df)
plt.title('Distribución de Sentimientos Antes del Balanceo')
plt.xlabel('Análisis de Sentimientos')
plt.ylabel('Cantidad')
plt.show()

# Ajustamos las categorías de sentimientos según los valores únicos
sentimientos_unicos = reviews_df['Analisis_sentimientos'].unique()
sentimiento_categorias = {}

for sentimiento in sentimientos_unicos:
    sentimiento_categorias[sentimiento] = reviews_df[reviews_df['Analisis_sentimientos'] == sentimiento]

# Buscamos el número máximo de muestras
max_samples = max([len(df) for df in sentimiento_categorias.values()])

# Función para remuestrear si el DataFrame no está vacío
def resample_if_not_empty(df, max_samples):
    if len(df) == 0:
        return df  # Retorna el DataFrame vacío si está vacío
    else:
        return resample(df, replace=True, n_samples=max_samples, random_state=42)

# Remuestreo de cada categoría
sentimiento_upsampled = []
for sentimiento, df in sentimiento_categorias.items():
    upsampled_df = resample_if_not_empty(df, max_samples)
    sentimiento_upsampled.append(upsampled_df)

# Combinamos las clases remuestreadas
reviews_balanced = pd.concat(sentimiento_upsampled)

# Verificamos la nueva distribución
print("\nDistribución de sentimientos después del balanceo:")
print(reviews_balanced['Analisis_sentimientos'].value_counts())

# Visualizamos la distribución de los sentimientos después del balanceo
plt.figure(figsize=(6, 4))
sns.countplot(x='Analisis_sentimientos', data=reviews_balanced)
plt.title('Distribución de Sentimientos Después del Balanceo')
plt.xlabel('Análisis de Sentimientos')
plt.ylabel('Cantidad')
plt.show()

# Calculamos el sentimiento promedio por juego
sentiment_mean = reviews_balanced.groupby('Id_de_post')['Analisis_sentimientos'].mean().reset_index()
sentiment_mean.rename(columns={'Analisis_sentimientos': 'Sentimiento_Promedio'}, inplace=True)

# Unimos al DataFrame de juegos
games_df['Nombre_App'] = games_df['Nombre_App'].astype(str)
sentiment_mean['Id_de_post'] = sentiment_mean['Id_de_post'].astype(str)
games_df = games_df.merge(sentiment_mean, left_on='Nombre_App', right_on='Id_de_post', how='left')
games_df['Sentimiento_Promedio'] = games_df['Sentimiento_Promedio'].fillna(0)
games_df.drop(columns=['Id_de_post'], inplace=True)

# Incorporamos el sentimiento en el contenido
games_df['Sentimiento_Promedio'] = games_df['Sentimiento_Promedio'].astype(str)
games_df['contenido'] = games_df['contenido'] + ' ' + games_df['Sentimiento_Promedio']

# Vectorización y similitud con sentimiento
tfidf_sentimiento = TfidfVectorizer(stop_words='english')
tfidf_matrix_sentimiento = tfidf_sentimiento.fit_transform(games_df['contenido'])
cosine_sim_sentimiento = cosine_similarity(tfidf_matrix_sentimiento, tfidf_matrix_sentimiento)

# Vectorización y similitud sin sentimiento
games_df['contenido_sin_sentimiento'] = games_df['Géneros_Juego'] + ' ' + games_df['Etiquetas_Juego']
games_df['contenido_sin_sentimiento'] = games_df['contenido_sin_sentimiento'].str.lower()
tfidf_sin_sentimiento = TfidfVectorizer(stop_words='english')
tfidf_matrix_sin_sentimiento = tfidf_sin_sentimiento.fit_transform(games_df['contenido_sin_sentimiento'])
cosine_sim_sin_sentimiento = cosine_similarity(tfidf_matrix_sin_sentimiento, tfidf_matrix_sin_sentimiento)

# Hacemos un mapeo de títulos a índices
indices = pd.Series(games_df.index, index=games_df['Título_Juego']).drop_duplicates()

# Función de recomendación con sentimiento
def get_recommendations(title, cosine_sim=cosine_sim_sentimiento):
    if title not in indices:
        print("El juego no se encuentra en la base de datos.")
        return pd.Series([], dtype=str)

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Puedes ajustar el número de recomendaciones
    game_indices = [i[0] for i in sim_scores]
    return games_df['Título_Juego'].iloc[game_indices].reset_index(drop=True)

# Función de recomendación sin sentimiento
def get_recommendations_sin_sentimiento(title, cosine_sim=cosine_sim_sin_sentimiento):
    if title not in indices:
        print("El juego no se encuentra en la base de datos.")
        return pd.Series([], dtype=str)

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Puedes ajustar el número de recomendaciones
    game_indices = [i[0] for i in sim_scores]
    return games_df['Título_Juego'].iloc[game_indices].reset_index(drop=True)

# Obtenemos recomendaciones
titulo_juego = 'Ironbound'  # Cambia el título del juego según tus necesidades

print(f"\nJuegos recomendados para '{titulo_juego}' incluyendo sentimiento:")
recomendaciones_con_sentimiento = get_recommendations(titulo_juego)
print(recomendaciones_con_sentimiento)

print(f"\nJuegos recomendados para '{titulo_juego}' sin incluir sentimiento:")
recomendaciones_sin_sentimiento = get_recommendations_sin_sentimiento(titulo_juego)
print(recomendaciones_sin_sentimiento)





#creamos datos de prueva

train_data, test_data = train_test_split(interactions, test_size=0.2, random_state=42)

#Generamos funciones del modelo
