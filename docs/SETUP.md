# Guía de Configuración y Despliegue (Setup)

Este documento detalla los pasos necesarios para replicar el entorno de ejecución del proyecto, tanto de forma local como en la nube.

## 1. Configuración del Entorno Local

### Requisitos Previos
(Para local se recomienda el uso .venv)
- Python 3.10+
- Node.js & npm (para el dashboard)
- Cuenta en MongoDB Atlas (para persistencia en la nube)

### Paso 1: Variables de Entorno
Crea un archivo `.env` en la raíz del proyecto con las siguientes variables:
```env
MONGODB_URI=mongodb+srv://<usuario>:<password>@cluster.mongodb.net/pdg_db
MONGODB_DB_NAME=pdg_db
```

### Paso 2: Instalación de Dependencias
```bash
# Instalar dependencias del backend
pip install -r requirements.txt

# Instalar dependencias del frontend
cd dashboard
npm install
cd ..
```

### Paso 3: Carga Inicial de Datos (GridFS)
Para cumplir con la regla de no subir datasets a GitHub, el sistema descarga los datos de Mongo. Debes subir tus archivos locales por única vez:
```bash
python scripts/upload_initial_datasets.py
```

### Paso 4: Ejecución
Abre dos terminales:
- **Backend**: `uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload`
- **Frontend**: `cd dashboard && npm run dev`

---

## 2. Despliegue en la Nube

El proyecto está diseñado para un despliegue desacoplado:

### Backend (Render)
- **Servicio**: Web Service (Docker).
- **URL**: [https://pdg.onrender.com](https://pdg.onrender.com)
- **Configuración**: El `Dockerfile` en la raíz maneja el ambiente. Se deben configurar las variables de entorno en el panel de Render.

### Frontend (Vercel)
- **Servicio**: Static Hosting.
- **URL**: [https://pdg-kappa.vercel.app](https://pdg-kappa.vercel.app)
- **Configuración**: El "Root Directory" debe ser `dashboard/`. La variable `VITE_API_URL` debe apuntar a la URL de Render.

---

## 3. Comandos Útiles del Pipeline (CLI)
Si deseas ejecutar pasos específicos desde la consola:
```bash
# Ejecutar todo el pipeline
python train_pipeline.py

# Ejecutar solo entrenamiento y evaluación
python train_pipeline.py --steps train,evaluate,compare
```

---
[⬅ Volver al README principal](../README.md)
