# Estructura del Proyecto

Este documento describe la organización del repositorio y la responsabilidad de cada componente en el sistema de predicción de riesgo quirúrgico.

## Directorios Principales

```text
PDG/
├── api/                  # Backend: Servidor FastAPI y lógica de Endpoints.
├── dashboard/            # Frontend: Aplicación React + Vite (Dashboard).
├── src/                  # Motor de ML: Scripts modulares del pipeline.
│   ├── db/               # Gestión de persistencia (MongoDB + GridFS).
│   ├── utils/            # Utilidades de carga de datos, logs y entrenamiento.
├── data/                 # Almacén de datos (Ignorado en Git, gestionado en Mongo).
│   ├── datasets/         # Datos crudos (.xlsx, .csv).
│   ├── processed/        # Datos limpios y particiones de entrenamiento.
│   └── results/          # Resultados de evaluación y JSON del dashboard.
├── models/               # Modelos entrenados (.pkl) (Ignorado en Git).
├── scripts/              # Scripts de utilidad (ej: carga inicial a Mongo).
├── config/               # Configuraciones YAML para entrenamiento y validación.
└── docs/                 # Documentación detallada del proyecto.
```

## Descripción de Componentes

### 1. Backend (`/api`)
Contiene el archivo `main.py`, que actúa como el orquestador entre el usuario y el motor de ML. Expone servicios para ejecutar el pipeline, consultar resultados históricos y realizar inferencias sobre nuevos pacientes.

### 2. Frontend (`/dashboard`)
Interfaz de usuario moderna que consume la API. Está desacoplada del backend, lo que permite su despliegue independiente. Permite la visualización de métricas de interpretabilidad (Feature Importance) y validación clínica (RACHS-1).

### 3. Motor de ML (`/src`)
Contiene la lógica modular del proyecto. Cada archivo representa una etapa del ciclo de vida del dato:
- `data_ingest.py`: Ingesta y versionado.
- `preprocess.py`: Limpieza y codificación.
- `balance.py`: Balanceo ADASYN.
- `train.py` & `evaluate.py`: Entrenamiento y métricas.
- `feature_importance.py`: Interpretabilidad.
- `rachs1.py`: Comparación clínica.

### 4. Persistencia (`/src/db`)
Implementa el patrón **MongoStore**, permitiendo que el proyecto sea "apátrida" (stateless). Los archivos pesados se guardan en GridFS, permitiendo que el backend en la nube recupere todo el estado sin necesidad de archivos locales en el repositorio.

---
[⬅ Volver al README principal](../README.md)
