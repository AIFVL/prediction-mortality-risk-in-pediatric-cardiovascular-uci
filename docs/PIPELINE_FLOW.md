# Flujo del Pipeline y Metodología Analítica

Este documento explica la transición técnica desde la experimentación en Notebooks hacia el sistema automatizado actual.

## 1. De Notebooks a Pipeline
El conocimiento exploratorio generado en los notebooks fue encapsulado en scripts modulares para garantizar la **reproducibilidad**:

- **Limpieza e Imputación**: De `1_data_cleansing_imputation.ipynb` a `src/preprocess.py`.
- **Balanceo (SMOTE/ADASYN)**: De `2_data_balancing_and_modeling_preparation.ipynb` a `src/balance.py`.
- **Entrenamiento**: De diversos notebooks de modelado a `src/train.py` y `src/models.py`.

## 2. Flujo de Comunicación (Dashboard -> API -> Pipeline)

El sistema opera bajo un flujo de orquestación asíncrona:

1.  **Dashboard (UI)**: El usuario solicita "Ejecutar Pipeline".
2.  **API (FastAPI)**: Recibe la petición y dispara `train_pipeline.py` como un proceso en segundo plano.
3.  **Pipeline (Engine)**: Ejecuta secuencialmente los 8 pasos.
4.  **Persistencia**: Cada paso genera archivos locales, pero el paso final sincroniza todo (modelos, métricas y JSON) con **MongoDB**.
5.  **Actualización**: El Dashboard consulta periódicamente el estado y, al finalizar, recupera los nuevos resultados desde la BD.

## 3. Detalle de los 8 Pasos del Pipeline

| Paso | Archivo | Entrada | Salida |
| :--- | :--- | :--- | :--- |
| **1. Ingest** | `data_ingest.py` | Excel (de Mongo/Local) | `pacientes_raw.csv` |
| **2. Preprocess** | `preprocess.py` | CSV Raw | Matrices $X, y$ limpias |
| **3. Balance** | `balance.py` | Matrices Raw | Matrices Balanceadas (ADASYN) |
| **4. Train** | `train.py` | Matrices | Modelos `.pkl` (SVM, XGB, etc.) |
| **5. Evaluate** | `evaluate.py` | Modelos + Test Set | `evaluation_summary.csv` |
| **6. Compare** | `compare.py` | Resúmenes | `model_comparison.csv` |
| **6b. Features** | `feature_importance.py` | Modelos | `feature_importance.json` |
| **7. Export** | `export_dashboard.py` | Todos los resultados | `dashboard_data.json` |
| **8. RACHS-1** | `rachs1.py` | Dashboard Data | Métricas comparativas finales |

## 4. Motor de Inferencia
Independiente del pipeline de entrenamiento, el archivo `src/inference.py` permite realizar predicciones individuales o masivas. Recupera el "Mejor Modelo" desde MongoDB y aplica las transformaciones de preprocesamiento en tiempo real para devolver el riesgo de mortalidad al clínico.

---
[⬅ Volver al README principal](../README.md)
