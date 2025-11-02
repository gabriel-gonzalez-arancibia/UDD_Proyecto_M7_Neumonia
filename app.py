# =============================================================================
# LIBRERÍAS NECESARIAS
# =============================================================================

import os
import gdown
import requests
import numpy as np
from flask                              import Flask, request, jsonify
from tensorflow.keras.models            import load_model
from tensorflow.keras.preprocessing     import image
from PIL                                import Image
import io

# --- Rutas ---
MODEL_DIR = "modelos_descargados"
os.makedirs(MODEL_DIR, exist_ok=True)

# ¡¡REEMPLAZA ESTAS URLS CON LAS TUYAS DE GITHUB RELEASES!!
MODEL_URLS = [
    'https://github.com/gabriel-gonzalez-arancibia/UDD_Proyecto_M7_Neumonia/releases/download/CNN/modelo_original.keras',
    'https://github.com/gabriel-gonzalez-arancibia/UDD_Proyecto_M7_Neumonia/releases/download/CNN/tuner_mejor_modelo_1.keras',
    'https://github.com/gabriel-gonzalez-arancibia/UDD_Proyecto_M7_Neumonia/releases/download/CNN/tuner_mejor_modelo_2.keras',
    'https://github.com/gabriel-gonzalez-arancibia/UDD_Proyecto_M7_Neumonia/releases/download/CNN/tuner_mejor_modelo_3.keras'
]

model_paths = [os.path.join(MODEL_DIR, os.path.basename(url)) for url in MODEL_URLS]

# --- Función de Descarga ---
def descargar_modelos():
    print("Verificando modelos...")
    for url, path in zip(MODEL_URLS, model_paths):
        if not os.path.exists(path):
            print(f"Descargando modelo desde {url}...")
            try:
                # Usar requests para descargar el archivo LFS de GitHub
                r = requests.get(url, allow_redirects=True)
                with open(path, 'wb') as f:
                    f.write(r.content)
                print(f"Modelo guardado en {path}")
            except Exception as e:
                print(f"Error al descargar {url}: {e}")
        else:
            print(f"Modelo {path} ya existe.")

# --- Carga de Modelos ---
descargar_modelos() # Llama a la función al iniciar la app
print("Cargando modelos del ensamble...")
ensemble_members = [load_model(path) for path in model_paths]
print(f"✅ {len(ensemble_members)} modelos del ensamble cargados en memoria.")

# ==============================================================================
# 1. INICIALIZAR LA APLICACIÓN FLASK
# ==============================================================================
app = Flask(__name__)

# ==============================================================================
# 2. CARGAR EL ENSAMBLE DE MODELOS EN MEMORIA
# ==============================================================================
print("Cargando modelos del ensamble... (esto puede tardar unos segundos)")

# Definimos las rutas a los modelos que guardamos
model_paths = [
    'modelo_original.keras',
    'ensemble_members/tuner_mejor_modelo_1.keras',
    'ensemble_members/tuner_mejor_modelo_2.keras',
    'ensemble_members/tuner_mejor_modelo_3.keras'
]

# Los cargamos en una lista
ensemble_members = []
for path in model_paths:
    try:
        ensemble_members.append(load_model(path))
    except Exception as e:
        print(f"Error cargando el modelo {path}: {e}")
        
print(f"{len(ensemble_members)} modelos del ensamble cargados en memoria.")

# Definimos las etiquetas de las clases
CLASS_LABELS = ['Bacterial', 'Normal', 'Viral']
IMG_SIZE = (128, 128) 

# ==============================================================================
# 3. FUNCIÓN DE PREPROCESAMIENTO
# ==============================================================================
def preprocess_image(img_bytes):
    """
    Toma los bytes de una imagen, la carga, la preprocesa
    y la prepara para el modelo.
    """
    # Cargar la imagen desde los bytes
    img = Image.open(io.BytesIO(img_bytes))
    
    # Asegurarse de que la imagen sea RGB (3 canales)
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    # Redimensionar la imagen al tamaño que espera el modelo
    img = img.resize(IMG_SIZE)
    
    # Convertir la imagen a un array de numpy
    img_array = image.img_to_array(img)
    
    # Reescalar los píxeles (igual que en el entrenamiento)
    img_array = img_array / 255.0
    
    # Expandir las dimensiones para que coincida con el (batch_size, height, width, channels)
    # (1, 128, 128, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    
    return img_batch

# ==============================================================================
# 4. FUNCIÓN DE PREDICCIÓN DEL ENSAMBLE
# ==============================================================================
def predecir_con_ensamble(img_batch):
    """
    Toma un lote de imagen preprocesada y devuelve la predicción
    promediada por el ensamble.
    """
    all_predictions = []
    for model in ensemble_members:
        # verbose=0 para evitar que imprima logs en la consola de la API
        preds = model.predict(img_batch, verbose=0)
        all_predictions.append(preds)
        
    # Promediar las predicciones
    ensemble_predictions_avg = np.mean(all_predictions, axis=0)
    
    # Obtener la clase final y la confianza
    final_class_index = np.argmax(ensemble_predictions_avg, axis=1)[0]
    confidence = np.max(ensemble_predictions_avg, axis=1)[0]
    
    # Mapear el índice a la etiqueta de texto
    predicted_label = CLASS_LABELS[final_class_index]
    
    return predicted_label, float(confidence) # Convertir confianza a float nativo

# ==============================================================================
# 5. DEFINIR EL ENDPOINT DE LA API
# ==============================================================================
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    # 1. Verificar que se haya enviado un archivo
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró la parte del archivo'}), 400
    
    file = request.files['file']
    
    # 2. Verificar que el archivo tenga un nombre
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
    
    try:
        # 3. Leer los bytes del archivo
        img_bytes = file.read()
        
        # 4. Preprocesar la imagen
        processed_image = preprocess_image(img_bytes)
        
        # 5. Realizar la predicción con el ensamble
        label, confidence = predecir_con_ensamble(processed_image)
        
        # 6. Devolver la respuesta en formato JSON
        return jsonify({
            'prediccion': label,
            'confianza': f"{confidence * 100:.2f}%"
        })
        
    except Exception as e:
        return jsonify({'error': f'Error durante el procesamiento: {str(e)}'}), 500

# ==============================================================================
# 6. EJECUTAR EL SERVIDOR
# ==============================================================================
if __name__ == '__main__':
    # host='0.0.0.0' permite que el servidor sea accesible desde tu red local
    app.run(host='0.0.0.0', port=5002)