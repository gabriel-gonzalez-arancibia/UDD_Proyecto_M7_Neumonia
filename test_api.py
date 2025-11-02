# Script: test_api.py
import requests
import os

# ---------------------------------------------------------------
# CONFIGURA ESTO:
# 1. Imagen de prueba local
PATH_A_IMAGEN_DE_PRUEBA = '/Users/gabrielgonzalez/Documents/Ciencia de Datos/M7 Técnicas Avanzadas/test/NORMAL/IM-0001-0001.jpeg'

# 2. La URL API local
API_URL = 'https://gabrielgonzalez-proyecto-neumonia-api.hf.space/predict'
# ---------------------------------------------------------------

# Verificamos que la imagen exista
if not os.path.exists(PATH_A_IMAGEN_DE_PRUEBA):
    print(f"Error: No se encuentra la imagen de prueba en: {PATH_A_IMAGEN_DE_PRUEBA}")
else:
    # Abrimos la imagen en modo binario
    with open(PATH_A_IMAGEN_DE_PRUEBA, 'rb') as f:
        files = {'file': (f.name, f, 'image/jpeg')}
        
        try:
            # Hacemos la solicitud POST a la API
            print(f"Enviando imagen a {API_URL}...")
            response = requests.post(API_URL, files=files)
            
            # Imprimimos la respuesta (el JSON) del servidor
            print("\nRespuesta del servidor:")
            print(response.json())
            
        except requests.exceptions.ConnectionError:
            print("\nError: No se pudo conectar a la API. ¿Está el servidor 'app.py' corriendo?")