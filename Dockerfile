# 1. Usar la imagen base de Python 
FROM python:3.10.11

# 2. Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Copiar el archivo de requerimientos e instalarlos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar el resto de tu código (app.py, modelos, etc.)
COPY . .

# 5. Exponer el puerto en el que correrá Gunicorn
EXPOSE 7860

# 6. Comando para iniciar el servidor
# Usamos Gunicorn para correr la variable 'app' que está en el archivo 'app.py'
# y le decimos que escuche en el puerto 7860
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860"]