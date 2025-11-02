import os
import numpy as np
import requests
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import warnings

# ==============================================================================
# 1. INICIALIZAR LA APLICACIÓN FLASK
# ==============================================================================
app = Flask(__name__)

# ==============================================================================
# 2. CARGAR EL MODELO MÁS LIGERO (59 MB)
# ==============================================================================
MODEL_DIR = "modelo_descargado_ligero"
os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# ¡¡ACCIÓN REQUERIDA!!
# Ve a tu página de GitHub Releases y copia la URL del modelo de 59MB.
# Pégala aquí:
# --------------------------------------------------------------------------
MODEL_URL = 'https://github.com/gabriel-gonzalez-arancibia/UDD_Proyecto_M7_Neumonia/releases/download/CNN/tuner_mejor_modelo_2.keras'
MODEL_PATH = os.path.join(MODEL_DIR, 'modelo_ligero.keras') # Guardamos como 'modelo_ligero.keras'

def descargar_modelo():
    print("Verificando modelo...")
    if not os.path.exists(MODEL_PATH):
        print(f"Descargando modelo ligero desde {MODEL_URL}...")
        try:
            r = requests.get(MODEL_URL, allow_redirects=True)
            with open(MODEL_PATH, 'wb') as f:
                f.write(r.content)
            print(f"Modelo guardado en {MODEL_PATH}")
        except Exception as e:
            print(f"Error al descargar {MODEL_URL}: {e}")
    else:
        print(f"Modelo {MODEL_PATH} ya existe.")

# --- Carga de Modelo ---
descargar_modelo()
print("Cargando modelo ligero en memoria...")
# Suprimimos warnings de Keras al cargar
warnings.filterwarnings('ignore', category=UserWarning)
model = load_model(MODEL_PATH)
warnings.filterwarnings('default', category=UserWarning)
print("✅ Modelo único cargado exitosamente.")

# Definimos las etiquetas de las clases
CLASS_LABELS = ['Bacterial', 'Normal', 'Viral']
IMG_SIZE = (128, 128)

# ==============================================================================
# 3. FUNCIÓN DE PREPROCESAMIENTO 
# ==============================================================================
def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

# ==============================================================================
# 4. FUNCIÓN DE PREDICCIÓN (Un solo modelo)
# ==============================================================================
def predecir(img_batch):
    preds = model.predict(img_batch, verbose=0)
    
    final_class_index = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds, axis=1)[0]
    predicted_label = CLASS_LABELS[final_class_index]
    
    return predicted_label, float(confidence)

# ==============================================================================
# 5. DEFINIR EL ENDPOINT DE LA API
# ==============================================================================
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró la parte del archivo'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
    
    try:
        img_bytes = file.read()
        processed_image = preprocess_image(img_bytes)
        
        # Llamamos a la nueva función de predicción simple
        label, confidence = predecir(processed_image)
        
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
    app.run(host='0.0.0.0', port=5002)