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
# 2. CARGAR EL ENSAMBLE LIGERO (2 Modelos)
# ==============================================================================
MODEL_DIR = "modelos_descargados_ensamble_ligero"
os.makedirs(MODEL_DIR, exist_ok=True)

# Sólo se utilizan 2 modelos por limitaciones de memoria en el servidor gratuito de Render
MODEL_URLS = [
    'https://github.com/gabriel-gonzalez-arancibia/UDD_Proyecto_M7_Neumonia/releases/download/CNN/modelo_original.keras',
    'https://github.com/gabriel-gonzalez-arancibia/UDD_Proyecto_M7_Neumonia/releases/download/CNN/tuner_mejor_modelo_1.keras'
]
model_paths = [os.path.join(MODEL_DIR, os.path.basename(url)) for url in MODEL_URLS]

def descargar_modelos():
    print("Verificando modelos...")
    for url, path in zip(MODEL_URLS, model_paths):
        if not os.path.exists(path):
            print(f"Descargando modelo desde {url}...")
            try:
                r = requests.get(url, allow_redirects=True)
                with open(path, 'wb') as f:
                    f.write(r.content)
                print(f"Modelo guardado en {path}")
            except Exception as e:
                print(f"Error al descargar {url}: {e}")
        else:
            print(f"Modelo {path} ya existe.")

# --- Carga de Modelos ---
descargar_modelos()
print("Cargando modelos del ensamble ligero...")
warnings.filterwarnings('ignore', category=UserWarning)
ensemble_members = [load_model(path) for path in model_paths]
warnings.filterwarnings('default', category=UserWarning)
print(f"✅ {len(ensemble_members)} modelos del ensamble cargados en memoria.")

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
# 4. FUNCIÓN DE PREDICCIÓN (Ensamble)
# ==============================================================================
def predecir_con_ensamble(img_batch):
    all_predictions = []
    for model in ensemble_members:
        preds = model.predict(img_batch, verbose=0)
        all_predictions.append(preds)
        
    # Promediar las predicciones
    ensemble_predictions_avg = np.mean(all_predictions, axis=0)
    
    final_class_index = np.argmax(ensemble_predictions_avg, axis=1)[0]
    confidence = np.max(ensemble_predictions_avg, axis=1)[0]
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
        
        label, confidence = predecir_con_ensamble(processed_image)
        
        return jsonify({
            'prediccion': label,
            'confianza': f"{confidence * 100:.2f}%"
        })
        
    except Exception as e:
        return jsonify({'error': f'Error durante el procesamiento: {str(e)}'}), 500

# ==============================================================================
# 6. EJECUTAR EL SERVIDOR (Sin cambios)
# ==============================================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)