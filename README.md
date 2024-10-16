Proyecto Individual: Sistema de Recomendación de Videojuegos en Steam

**Descripción del Proyecto

Este proyecto tiene como objetivo desarrollar un MVP de un sistema de recomendación de videojuegos en Steam, en el contexto de un entorno de Machine Learning Operations (MLOps). Como MLOps Engineer, se encargó de todo el ciclo de vida del proyecto, desde la recolección y limpieza de datos hasta el despliegue de un modelo de machine learning y la creación de una API RESTful utilizando FastAPI. El modelo de recomendación permitirá ofrecer a los usuarios de Steam videojuegos basados en sus interacciones y preferencias.
Desafíos:

    Datos con demaciados errores y no procesados, con anidamiento y falta de automatización en los procesos.
    Transformaciones de datos mínimas para optimizar la API y el modelo.
    Implementación de análisis de sentimientos en las reseñas de los usuarios.
    Desarrollo de endpoints para la consulta de datos y recomendaciones.
    Entrenamiento y despliegue de un modelo de machine learning (item-item o user-item).
    Deployment del sistema en un servicio accesible desde la web.

Propuesta de Trabajo

El sistema de recomendación sigue los siguientes pasos para completar su ciclo de vida:

    Transformación de Datos:
        El dataset se procesa para eliminar columnas innecesarias y optimizar el rendimiento del sistema.
        Se crea una nueva columna 'sentiment_analysis' a partir de análisis de sentimientos aplicado a las reseñas de los usuarios, categorizándolas como:
            0: Reseña negativa
            1: Reseña neutral o sin texto
            2: Reseña positiva

    Feature Engineering:
        Las reseñas de usuarios fueron procesadas para añadir la columna de análisis de sentimientos, facilitando el entrenamiento del modelo y las consultas futuras.

    Desarrollo de la API: Se desarrollaron los siguientes endpoints para ser consumidos por usuarios y otros sistemas:

        GET /developer/{desarrollador}: Retorna el número de items y el porcentaje de contenido gratuito de cada desarrollador por año.

        GET /userdata/{user_id}: Proporciona información sobre el usuario, incluyendo cantidad de dinero gastado, porcentaje de recomendación y número de items.

        GET /UserForGenre/{genero}: Devuelve el usuario con más horas jugadas en un género específico y un resumen por año de las horas jugadas.

        GET /best_developer_year/{año}: Muestra los tres desarrolladores más recomendados en un año específico, basándose en las reseñas de los usuarios.

        GET /developer_reviews_analysis/{desarrolladora}: Retorna un análisis de las reseñas según el desarrollador, mostrando un conteo de reseñas positivas y negativas.

        GET /recomendacion_juego/{product_id}: Sistema de recomendación basado en la similitud entre juegos (item-item), que devuelve una lista de juegos similares a un producto dado.

        POST /recomendacion_usuario/{user_id}: Sistema de recomendación basado en la similitud entre usuarios (user-item), que sugiere juegos que a usuarios similares les han gustado.

    Exploratory Data Analysis (EDA): Un análisis exploratorio de los datos fue realizado para identificar patrones, detectar outliers y anomalías, y generar nubes de palabras para visualizar las palabras más frecuentes en los títulos de juegos.

    Modelo de Machine Learning: Se implementó un modelo de recomendación basado en:
        Item-Item: Utilizando la similitud de coseno, este modelo recomienda videojuegos similares a uno dado.
        User-Item: Se recomienda a los usuarios juegos que les gustaron a otros usuarios con intereses similares.

    Deployment: El sistema fue desplegado utilizando Render o Railway, permitiendo el acceso a la API desde cualquier dispositivo conectado a internet.

Estructura del Proyecto

