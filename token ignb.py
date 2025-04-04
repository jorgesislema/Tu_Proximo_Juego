import requests
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')

url = 'https://id.twitch.tv/oauth2/token'
params = {
    'client_id': client_id,
    'client_secret': client_secret,
    'grant_type': 'client_credentials'
}

response = requests.post(url, params=params)

if response.status_code == 200:
    access_token = response.json()['access_token']
    print("Access Token:", access_token)
else:
    print("Error al obtener el token:", response.text)
