# PDG — Walkthrough de configuración y ejecución

## Estructura de datos esperada

```
PDG/
├── data/
│   ├── datasets/
│   │   └── dataset_inicial.xlsx   ← tu dataset original aquí
│   ├── processed/
│   │   └── csv/                   ← generado automáticamente por el pipeline
│   └── results/
│       └── dashboard_data.json    ← generado automáticamente por el pipeline
├── models/                        ← generado automáticamente por el pipeline
├── src/                           ← módulos del pipeline
├── api/
│   └── main.py                    ← servidor FastAPI (puerto 8000)
├── dashboard/                     ← frontend React/Vite (puerto 5000)
└── train_pipeline.py              ← CLI del pipeline
```

---

## Paso 1 — Colocar el dataset

Copia tu archivo Excel al directorio correcto:

```bash
# Si tienes el archivo en otro lugar, cópialo así:
cp /ruta/a/tu/dataset_inicial.xlsx data/datasets/dataset_inicial.xlsx
```

El pipeline lee `data/datasets/dataset_inicial.xlsx` por defecto. No se necesita configurar nada más.

---

## Paso 2 — Crear las carpetas necesarias

Ejecuta esto una sola vez para preparar el entorno:

```bash
mkdir -p data/datasets data/processed/csv data/processed/pickle data/reports data/results models logs
```

---

## Paso 3 — Instalar dependencias Python

```bash
pip install fastapi uvicorn pandas numpy scikit-learn imbalanced-learn xgboost openpyxl
```

---

## Paso 4 — Instalar dependencias del dashboard

```bash
cd dashboard && npm install && cd ..
```

---

## Paso 5 — Iniciar el backend (API)

Abre una terminal y ejecuta:

```bash
python -m uvicorn api.main:app --host localhost --port 8000 --reload
```

La API queda corriendo en `http://localhost:8000`.  
Endpoints disponibles:
- `GET  /api/health` → verifica que el servidor está activo
- `GET  /api/results` → devuelve los resultados del último pipeline
- `POST /api/pipeline/run` → lanza el pipeline desde el dashboard
- `GET  /api/pipeline/status` → estado del pipeline (idle / running / success / error)
- `GET  /api/pipeline/log` → log en tiempo real del pipeline

---

## Paso 6 — Ejecutar el pipeline (CLI)

Abre **otra terminal** y ejecuta:

```bash
# Ejecución completa con todos los modelos (por defecto usa dataset_inicial.xlsx)
python train_pipeline.py

# Con modelos específicos
python train_pipeline.py --models logistic_regression,random_forest

# Con búsqueda de hiperparámetros (más lento)
python train_pipeline.py --use-gridsearch

# Con un dataset diferente
python train_pipeline.py --input-path data/datasets/otro_dataset.xlsx

# Solo algunos pasos (ej: re-entrenar y re-exportar sin re-ingestar)
python train_pipeline.py --steps preprocess,balance,train,evaluate,compare,export
```

El pipeline corre estos 7 pasos en orden:
1. **Ingest** — lee el xlsx y lo guarda como CSV en `data/datasets/`
2. **Preprocess** — limpia columnas, codifica variables, deriva la variable objetivo (3 clases), divide 70/30
3. **Balance** — aplica SMOTE al conjunto de entrenamiento
4. **Train** — entrena cada modelo en versión RAW y SMOTE
5. **Evaluate** — calcula métricas en el test set (accuracy, F1, ROC-AUC, kappa, etc.)
6. **Compare** — consolida los resultados de todos los modelos
7. **Export** — genera `data/results/dashboard_data.json` para el dashboard

---

## Paso 7 — Iniciar el dashboard

Abre **otra terminal** y ejecuta:

```bash
cd dashboard && npm run dev
```

El dashboard queda disponible en `http://localhost:5000`.

Si el pipeline ya corrió, al abrir el dashboard verás automáticamente:
- El **mejor modelo** según Balanced Accuracy
- La **distribución de clases** (RAW vs SMOTE)
- La **tabla comparativa** de todos los modelos
- Los **gráficos de métricas**
- La **matriz de confusión**
- Las **probabilidades promedio por clase**

---

## Paso 8 — (Opcional) Ejecutar el pipeline desde el dashboard

En el dashboard, haz clic en **"Ejecutar Pipeline"**.  
El backend lanzará el pipeline en segundo plano y podrás ver el log en tiempo real.  
Al finalizar, haz clic en **"Actualizar"** para ver los nuevos resultados.

---

## Resumen de comandos (copiar y pegar)

```bash
# Terminal 1 — API Backend
python -m uvicorn api.main:app --host localhost --port 8000 --reload

# Terminal 2 — Pipeline
python train_pipeline.py

# Terminal 3 — Dashboard
cd dashboard && npm run dev
```

---

## Variable objetivo (3 clases)

La columna `Mortalidad` se deriva automáticamente de las columnas originales del Excel:

| Clase | Etiqueta       | Condición                                        |
|-------|----------------|--------------------------------------------------|
| 0     | no murio       | `MORTALIDAD GENERAL` = 0                         |
| 1     | murio (>30d)   | `MORTALIDAD GENERAL` = 1 y `MORTALIDAD A 30 DÍAS` = 0 |
| 2     | murio (<30d)   | `MORTALIDAD GENERAL` = 1 y `MORTALIDAD A 30 DÍAS` = 1 |

---

## Modelos disponibles

| Nombre                | Flag CLI                  |
|-----------------------|---------------------------|
| Regresión Logística   | `logistic_regression`     |
| Random Forest         | `random_forest`           |
| SVM                   | `svm`                     |
| XGBoost               | `xgboost`                 |
