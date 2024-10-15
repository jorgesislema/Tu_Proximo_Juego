from fastapi import FastAPI
import pandas as pd

app = FastAPI()

# Cargar los datos
games_df = pd.read_csv('data/output_steam_games.csv')
reviews_df = pd.read_csv('data/reseñas.csv')

# --- Endpoint 1: Cantidad de ítems y porcentaje de contenido Free por año según desarrollador ---
@app.get("/developer/")
def developer(desarrollador: str):
    # Filtrar los juegos del desarrollador
    dev_games = games_df[games_df['Desarrollador'] == desarrollador]
    
    # Agrupar por año
    dev_games['Año'] = pd.to_datetime(dev_games['Fecha_Lanzamiento'], errors='coerce').dt.year
    yearly_stats = dev_games.groupby('Año').agg(
        Cantidad_Items=('Nombre_App', 'count'),
        Contenido_Free=('Precio', lambda x: (x == 'Free to Play').mean() * 100)
    ).reset_index()

    return yearly_stats.to_dict(orient="records")


# --- Endpoint 2: Dinero gastado por usuario, porcentaje de recomendación y cantidad de ítems ---
@app.get("/userdata/")
def userdata(User_id: str):
    user_reviews = reviews_df[reviews_df['Id_usuario'] == User_id]
    
    # Calcular dinero gastado, cantidad de items y porcentaje de recomendación
    total_spent = user_reviews['Precio'].sum()
    recommended_pct = (user_reviews['Recomendado'] == True).mean() * 100
    total_items = user_reviews['Id_de_post'].nunique()

    return {
        "Usuario": User_id,
        "Dinero gastado": f"{total_spent} USD",
        "% de recomendación": f"{recommended_pct:.2f}%",
        "Cantidad de items": total_items
    }


# --- Endpoint 3: Usuario con más horas jugadas para un género ---
@app.get("/userforgenre/")
def user_for_genre(genero: str):
    # Filtrar los juegos por género
    genre_games = games_df[games_df['Géneros_Juego'].str.contains(genero, na=False)]

    # Combinar reseñas con juegos para obtener horas jugadas
    merged_df = pd.merge(reviews_df, genre_games, left_on='Id_de_post', right_on='Nombre_App', how='inner')

    # Agrupar por usuario para obtener el total de horas jugadas
    user_playtime = merged_df.groupby('Id_usuario')['Horas_Jugadas'].sum().reset_index()

    # Obtener el usuario con más horas jugadas
    top_user = user_playtime.loc[user_playtime['Horas_Jugadas'].idxmax()]

    # Acumulación de horas jugadas por año de lanzamiento
    merged_df['Año'] = pd.to_datetime(merged_df['Fecha_Lanzamiento'], errors='coerce').dt.year
    yearly_hours = merged_df.groupby('Año')['Horas_Jugadas'].sum().reset_index()

    return {
        "Usuario con más horas jugadas para Género X": top_user['Id_usuario'],
        "Horas jugadas": yearly_hours.to_dict(orient="records")
    }


# --- Endpoint 4: Top 3 desarrolladores con juegos más recomendados por año ---
@app.get("/best_developer_year/")
def best_developer_year(año: int):
    # Filtrar juegos por año
    year_games = games_df[pd.to_datetime(games_df['Fecha_Lanzamiento'], errors='coerce').dt.year == año]

    # Combinar con las reseñas para obtener las recomendaciones
    merged_df = pd.merge(reviews_df, year_games, left_on='Id_de_post', right_on='Nombre_App', how='inner')

    # Filtrar solo los recomendados
    recommended_games = merged_df[merged_df['Recomendado'] == True]

    # Agrupar por desarrollador y contar las recomendaciones
    top_developers = recommended_games.groupby('Desarrollador').size().reset_index(name='Cantidad_Recomendaciones')

    # Obtener los top 3 desarrolladores
    top_3 = top_developers.nlargest(3, 'Cantidad_Recomendaciones')

    return top_3.to_dict(orient="records")


# --- Endpoint 5: Análisis de reseñas por desarrollador ---
@app.get("/developer_reviews_analysis/")
def developer_reviews_analysis(desarrolladora: str):
    # Filtrar los juegos del desarrollador
    dev_games = games_df[games_df['Desarrollador'] == desarrolladora]

    # Combinar con las reseñas
    merged_df = pd.merge(reviews_df, dev_games, left_on='Id_de_post', right_on='Nombre_App', how='inner')

    # Contar reseñas positivas y negativas
    sentiment_counts = merged_df.groupby('Analisis_sentimientos').size().to_dict()

    return {desarrolladora: {'Negative': sentiment_counts.get(-1, 0), 'Positive': sentiment_counts.get(1, 0)}}
