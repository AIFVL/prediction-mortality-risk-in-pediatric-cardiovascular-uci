FROM python:3.10-slim

WORKDIR /app

# Instalamos dependencias del sistema necesarias para librerías de ML y compilación
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Aseguramos que pip esté actualizado para manejar mejor los wheels
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copiamos los requerimientos e instalamos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el código del proyecto
COPY . .

# Exponemos el puerto de FastAPI
EXPOSE 8000

# Comando para iniciar la API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]