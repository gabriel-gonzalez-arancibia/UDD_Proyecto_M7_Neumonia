# Proyecto Final: Detección de Neumonía con CNN (Módulo 7)

Este proyecto corresponde a la entrega final del Módulo 7 del Bootcamp de Ciencia de Datos e Inteligencia Artificial UDD. El objetivo es desarrollar un modelo de *Deep Learning* capaz de clasificar imágenes de radiografías de tórax en tres categorías: **Normal**, **Neumonía Bacterial** y **Neumonía Viral**.

El proyecto abarca el ciclo de vida completo de la ciencia de datos, desde el análisis exploratorio (EDA) y preprocesamiento, hasta el entrenamiento de modelos, la optimización de hiperparámetros y el despliegue final en una API REST.

---
## 🗂️ Archivos del Repositorio

* **`GG-ProyectoM7.ipynb`**: Jupyter Notebook con todo el proceso de análisis, preprocesamiento, entrenamiento y evaluación.
* **`app.py`**: El script de la API REST (creada con Flask) que carga el ensamble de modelos y sirve las predicciones.
* **`test_api.py`**: Script cliente para enviar una imagen de prueba a la API y recibir una predicción.
* **`requirements.txt`**: Lista de todas las dependencias de Python necesarias para replicar el proyecto.
* **`modelo_original.keras`**: El modelo base original, que sirvió como referencia.
* **`ensemble_members/`**: Carpeta que contiene los 3 mejores modelos encontrados durante el *tuning* con KerasTuner.
* **`.gitignore`**: Especifica los archivos que no se deben subir al repositorio (como los datos).
* **`.gitattributes`**: Configuración de Git LFS para manejar los archivos de modelo `.keras`.

---
## 📊 Resultados del Modelo Final

El modelo final implementado es un **ensamble de 4 CNNs** (el modelo base original + los 3 mejores del *tuning*), cuyas predicciones se promedian para obtener un resultado más robusto y con menor varianza.

### Métricas Cuantitativas

El ensamble logró una **Precisión General del 77.4%** en el conjunto de prueba (datos nunca vistos).

El rendimiento detallado por clase es el siguiente:

| Clase | Precisión (Precision) | Sensibilidad (Recall) | F1-Score |
| :--- | :--- | :--- | :--- |
| **Bacterial** | 0.81 | 0.79 | 0.80 |
| **Normal** | 0.82 | 0.96 | 0.88 |
| **Viral** | 0.64 | 0.56 | 0.60 |

### Análisis de Métricas Visuales

**(Aquí es donde puedes arrastrar y soltar tus imágenes `image_18c7a6.png` y `image_1920dd.png` en el editor de GitHub)**

**Matriz de Confusión:**
La matriz confirma que el modelo es **excepcionalmente bueno para identificar la clase 'Normal'** (Recall de 0.96). El principal desafío es la confusión entre los dos tipos de neumonía, donde 137 casos de 'Viral' fueron clasificados erróneamente como 'Bacterial', un sesgo consistente con el desbalance de clases del dataset original.

**Curvas ROC / AUC:**
El modelo muestra una fuerte capacidad de discriminación. El **AUC para 'Normal' es casi perfecto (0.98)**. 'Bacterial' (0.88) y 'Viral' (0.81) también muestran un buen rendimiento, confirmando que 'Viral' es la clase más difícil de distinguir para el modelo.

---
## 🚀 Cómo Usar la API (Localmente)

Para replicar y probar la API en un entorno local:

### 1. Instalación
Clona el repositorio y crea un entorno virtual. Luego, instala las dependencias:

Bash
git clone [https://github.com/gabriel-gonzalez-arancibia/UDD_Proyecto_M7_Neumonia.git](https://github.com/gabriel-gonzalez-arancibia/UDD_Proyecto_M7_Neumonia.git)
cd UDD_Proyecto_M7_Neumonia
pip install -r requirements.txt

Nota para macOS (Apple Silicon): Este proyecto se desarrolló con tensorflow-macos==2.16.2 y tensorflow-metal==1.2.0. Es posible que requirements.txt necesite un ajuste manual para estas librerías.

2. Ejecutar el Servidor
En un terminal, inicia el servidor Flask:

Bash

python app.py
El servidor se iniciará y cargará los 4 modelos en memoria. Estará listo en http://127.0.0.1:5001.

3. Probar la API
Abre un segundo terminal y ejecuta el script cliente test_api.py. (Asegúrate de que la variable PATH_A_IMAGEN_DE_PRUEBA dentro del script apunte a una imagen real).

Bash

python test_api.py
Respuesta esperada:

{
  "confianza": "87.13%",
  "prediccion": "Normal"
}