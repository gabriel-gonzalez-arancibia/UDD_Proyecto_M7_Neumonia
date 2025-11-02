---
title: Proyecto Deteccion de Neumonia
sdk: docker
app_port: 7860
---

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

<img width="777" height="699" alt="CM" src="https://github.com/user-attachments/assets/8b0be78e-ea18-4863-98a6-5ab3851469ed" />

**Matriz de Confusión:**
La matriz confirma que el modelo es **excepcionalmente bueno para identificar la clase 'Normal'** (Recall de 0.96). El principal desafío es la confusión entre los dos tipos de neumonía, donde 137 casos de 'Viral' fueron clasificados erróneamente como 'Bacterial', un sesgo consistente con el desbalance de clases del dataset original.

<img width="1004" height="699" alt="ROC" src="https://github.com/user-attachments/assets/5c5307c7-21d6-43a2-b4b1-1faee5201631" />

**Curvas ROC / AUC:**
El modelo muestra una fuerte capacidad de discriminación. El **AUC para 'Normal' es casi perfecto (0.98)**. 'Bacterial' (0.88) y 'Viral' (0.81) también muestran un buen rendimiento, confirmando que 'Viral' es la clase más difícil de distinguir para el modelo.

---

## 🚀 Cómo Probar la API Pública (Despliegue)

La API final se ha desplegado en **Hugging Face Spaces**, que proporciona la RAM gratuita necesaria (16GB) para ejecutar el ensamble completo de 4 modelos.

**URL Pública y Permanente de la API:**
`https://gabrielgonzalez-proyecto-neumonia-api.hf.space/predict`

### Prueba de la API Pública

Se puede probar la API enviando una solicitud POST con una imagen.

1.  Clona el repositorio.
2.  Instala las dependencias: `pip install -r requirements.txt`.
3.  Ejecuta el script `test_api.py`. (La URL ya está configurada para apuntar a la API de Hugging Face).

    ```bash
    python test_api.py
    ```
**Respuesta esperada:**
```json
{
  "confianza": "64.48%",
  "prediccion": "Normal"
}
```

🛠️ Cómo Replicar la Configuración Local
Si se desea ejecutar el servidor localmente:

1. Instalación
Clona el repositorio e instala las dependencias:

```Bash

git clone [https://github.com/gabriel-gonzalez-arancibia/UDD_Proyecto_M7_Neumonia.git](https://github.com/gabriel-gonzalez-arancibia/UDD_Proyecto_M7_Neumonia.git)
cd UDD_Proyecto_M7_Neumonia
pip install -r requirements.txt
```

Nota para macOS (Apple Silicon): Este proyecto se desarrolló con tensorflow-macos==2.16.2 y tensorflow-metal==1.2.0.

2. Ejecutar el Servidor Local
En un terminal, inicia el servidor Flask:


```Bash

python app.py
```
El servidor se iniciará en http://127.0.0.1:7860.

3. Probar la API Abre un segundo terminal y ejecuta el script cliente test_api.py. (Asegúrate de que la variable PATH_A_IMAGEN_DE_PRUEBA dentro del script apunte a una imagen real).

```Bash
python test_api.py
```

Respuesta esperada:

```json
{
  "confianza": "87.13%",
  "prediccion": "Normal"
}
```
