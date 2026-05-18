# Pipelines del proyecto PDG (mortalidad postoperatoria)

Este documento describe la organización y el funcionamiento de los scripts ubicados en `src/`,
su relación con los notebooks existentes y el flujo completo necesario para entrenar y evaluar
modelos de predicción de mortalidad postoperatoria.

## Correspondencia notebooks ↔ scripts

- `notebooks/0_Exploratory_data_analysis.ipynb`
  - Notebook de análisis exploratorio de datos (EDA). No se traduce directamente a un
    pipeline automatizado y se utiliza principalmente con fines descriptivos.

- `notebooks/1_data_cleansing_imputation.ipynb`
  - Contiene la lógica de limpieza, imputación y selección/transformación de variables.
  - Su contenido se migra progresivamente a:
    - `src/preprocess.py` → función `basic_cleaning(df: pd.DataFrame) -> pd.DataFrame`.

- `notebooks/2_data_balancing_and_modeling_preparation.ipynb`
  - Desarrolla la lógica de preparación de matrices de características (X), variable objetivo (y)
    y balanceo con SMOTE.
  - Esta lógica se refleja en:
    - `src/preprocess.py` → creación de `X_train_raw.csv`, `X_test_raw.csv`, `y_train_raw.csv`, `y_test_raw.csv`.
    - `src/balance.py` → creación de `X_train_balanced.csv`, `y_train_balanced.csv` y `metadatos_balanced.json`.

- `notebooks/modeling/training/*_training.ipynb`
  - Implementan el entrenamiento de cada modelo y, en algunos casos, procedimientos de
    búsqueda de hiperparámetros (grid search).
  - Su lógica se sintetiza en:
    - `src/models.py` → definición de modelos y grids básicos.
    - `src/train.py` → entrenamiento simple o mediante `GridSearchCV` para varios modelos.

- `notebooks/modeling/evaluation/*_evaluation.ipynb`
  - Contienen el cálculo detallado de métricas, curvas ROC/PR, matrices de confusión y
    comparaciones entre configuraciones.
  - Esta funcionalidad se transfiere a:
    - `src/evaluate.py` → cálculo de métricas globales y AUC multiclass.
    - `src/compare.py` → resumen combinado de resultados.
    - Funciones adicionales en `src/evaluate.py` para la generación de figuras pueden añadirse
      en etapas posteriores, reutilizando la lógica de estos notebooks.

## Descripción de cada archivo en `src/`

### `src/config.py`

Configuración global del proyecto:

- Define rutas base:
  - `DATASETS_DIR` → `data/datasets/`
  - `PROCESSED_DIR` → `data/processed/`
  - `RESULTS_DIR` → `data/results/`
  - `MODELS_DIR` → `models/`
- Define:
  - `TARGET_COLUMN = "Mortalidad"`
  - `RANDOM_STATE` y `TEST_SIZE` (coherentes con tu `metadatos.json`).
  - `DEFAULT_MODELS` → lista de modelos a entrenar si no especificas otros.

### `src/io_utils.py`

Conjunto de funciones de utilidad para operaciones de entrada/salida:

- `ensure_dir(path)` → crea carpetas si no existen.
- `read_csv(path)` / `write_csv(df, path)` → lectura/escritura de CSV.
- `file_hash(path)` → hash SHA‑256 de un archivo (para registrar versiones).
- `save_json(data, path)` / `load_json(path)` → guardar/cargar JSON.
- `build_versioned_name(base_name, version_tag)` → compone nombres versionados.

### `src/data_ingest.py`

Módulo de ingesta y versionado de nuevos datos crudos.

- `run_ingest(input_path: str, version_tag: Optional[str]) -> Path`
  - Lee un CSV externo de pacientes.
  - Lo copia a `data/datasets/` con un nombre tipo `pacientes_raw_<timestamp>.csv`
    (o usando `version_tag` si lo pasas).
  - Crea un JSON `pacientes_raw_<...>_metadata.json` con:
    - ruta de origen y de destino, nº filas/columnas, hash y timestamp.
  - Devuelve la ruta del CSV guardado, que usará el siguiente paso.

### `src/preprocess.py`

Módulo de limpieza, imputación, ingeniería de variables y particionado entrenamiento/prueba.

- `basic_cleaning(df) -> pd.DataFrame`
  - Actualmente funciona como *placeholder* (devuelve una copia del `DataFrame`).
  - Está diseñado para albergar la lógica de limpieza consolidada procedente de
    `1_data_cleansing_imputation.ipynb` (por ejemplo, manejo de valores perdidos,
    recodificaciones, creación de nuevas variables clínicas y escalados).

- `run_preprocessing(raw_path: str, version_tag: Optional[str])`
  - Lee el CSV raw desde `raw_path`.
  - Verifica que exista la columna objetivo `Mortalidad`.
  - Aplica `basic_cleaning`.
  - Separa `X` (todas las columnas menos `Mortalidad`) e `y` (`Mortalidad`).
  - Hace `train_test_split` estratificado usando `TEST_SIZE` y `RANDOM_STATE`.
  - Guarda en `data/processed/csv/`:
    - `X_train_raw.csv`, `X_test_raw.csv`, `y_train_raw.csv`, `y_test_raw.csv`.
  - Crea `data/processed/metadatos_generated.json` con shapes, test_size, etc.

### `src/balance.py`

Módulo de balanceo de clases en el conjunto de entrenamiento mediante SMOTE.

- `run_balance(version_tag: Optional[str])`
  - Carga `X_train_raw.csv` y `y_train_raw.csv`.
  - Aplica `SMOTE` (de `imblearn`) con el `RANDOM_STATE` definido.
  - Guarda en `data/processed/csv/`:
    - `X_train_smote.csv` y `y_train_smote.csv`.
  - Crea `data/processed/metadatos_smote.json` con:
    - shapes antes/después y conteo de clases, todo similar a tu `metadatos.json`.

### `src/models.py`

Definición de modelos de aprendizaje automático y rejillas de hiperparámetros.

- `get_model(model_name, random_state, class_weight)`
  - `"logistic_regression"` → `LogisticRegression` multinomial.
  - `"random_forest"` → `RandomForestClassifier`.
  - `"svm"` → `SVC` con `probability=True`.
  - `"xgboost"` → `XGBClassifier` (si tienes `xgboost` instalado).

- `get_param_grid(model_name)`
  - Devuelve diccionarios de grids sencillos para `GridSearchCV`, por ejemplo:
    - para `random_forest`: nº de árboles y profundidad.
  - Equivalente simplificado a los grids que has probado en los notebooks.

### `src/train.py`

Entrenamiento de todos los modelos definidos para los conjuntos `raw` y `balanced`.

- `_load_train_data(dataset_type)` → helper interno que lee
  `X_train_<dataset_type>.csv` y `y_train_<dataset_type>.csv`.

- `run_training(model_names=None, dataset_types=("raw","balanced"), use_gridsearch=False, ...)`
  - Si `model_names` es `None`, usa `DEFAULT_MODELS` de `config.py`.
  - Para cada `dataset_type` que exista (`raw`, `balanced`):
    - Crea un `Pipeline(StandardScaler() → modelo)`.
    - Si `use_gridsearch=True` y hay grid definido:
      - Ejecuta `GridSearchCV` con `scoring="recall_macro"` (por defecto).
      - Guarda resultados de CV en:
        - `data/results/<modelo>/gridsearch_results_<dataset_type>.csv`
    - En todos los casos guarda el mejor estimador como:
      - `models/<modelo>/<dataset_type>/model.pkl`.

### `src/evaluate.py`

Evaluación sobre el conjunto de prueba (actualmente el conjunto *raw*).

- `_load_test_data()` → lee `X_test_raw.csv` y `y_test_raw.csv`.
- `_load_model(model_name, dataset_type)` → carga `models/<modelo>/<dataset_type>/model.pkl`.

- `run_evaluation(model_names=None, dataset_types=("raw","smote"))`
  - Para cada combinación `modelo`/`dataset_type` con modelo entrenado:
    - Predice en `X_test_raw`.
    - Calcula:
      - `accuracy`, `balanced_accuracy`.
      - `macro_f1`, `weighted_f1` a partir de `classification_report`.
      - `cohen_kappa_score`, `matthews_corrcoef`.
      - Si hay probabilidades, `roc_auc_score` multiclass OVR macro.
  - Guarda un `evaluation_summary_<timestamp>.csv` en `data/results/` con una fila
    por combinación modelo/dataset.

> Nota: para reproducir exactamente todas las figuras de los notebooks
> (matrices de confusión, curvas ROC/PR por clase), pueden añadirse funciones
> específicas en este módulo; en la versión actual se prioriza la generación de
> tablas numéricas que den soporte a un futuro dashboard.

### `src/compare.py`

Combinación de todos los resúmenes de evaluación generados en distintas ejecuciones.

- `run_comparison()`
  - Busca todos los `evaluation_summary_*.csv` en `data/results/`.
  - Los concatena y guarda `data/results/model_comparison.csv`.
  - Esto te permite tener una tabla única de comparación para el dashboard.

### `src/predict.py`

Generación de predicciones en modo batch para nuevos pacientes.

- `run_predict(input_path, model_name, dataset_type="raw", output_name=None)`
  - Carga un CSV de entrada con nuevos pacientes (ya con las mismas columnas
    que usas en entrenamiento).
  - Carga `models/<modelo>/<dataset_type>/model.pkl`.
  - Genera:
    - columna `prediction`.
    - columnas `prob_class_0`, `prob_class_1`, ... si el modelo tiene `predict_proba`.
  - Guarda el resultado en `data/results/predictions/` con un nombre que incluye
    modelo, tipo de dataset y timestamp (a menos que pases `output_name`).

### `src/train_pipeline.py`

Módulo orquestador principal del flujo (ejecución de extremo a extremo mediante un único comando).

Argumentos clave:

- `--input-path` → ruta a un CSV externo para ingesta.
- `--raw-path` → ruta a un CSV ya situado en `data/datasets/` (para saltarte ingesta).
- `--version-tag` → etiqueta opcional para identificar la corrida.
- `--steps` → lista coma‑separada de pasos a ejecutar. Por defecto:
  - `ingest,preprocess,balance,train,evaluate,compare`.
- `--models` → lista de modelos a entrenar, por ejemplo:
  - `--models logistic_regression,random_forest`.
- `--use-gridsearch` → si lo incluyes, activa `GridSearchCV`.

Flujo:

1. Si `steps` contiene `ingest`, se llama a `run_ingest` con `--input-path`.
2. Si `steps` contiene `preprocess`, se llama a `run_preprocessing` usando el
   raw de ingest (o `--raw-path` si no se hizo ingest).
3. Si `steps` contiene `balance`, se llama a `run_balance`.
4. Si `steps` contiene `train`, se llama a `run_training`.
5. Si `steps` contiene `evaluate`, se llama a `run_evaluation`.
6. Si `steps` contiene `compare`, se llama a `run_comparison`.

## Cómo probar el pipeline

### 1. Instalar dependencias

En el entorno virtual asociado al proyecto PDG:

```bash
pip install scikit-learn imbalanced-learn xgboost
```

(En caso de que ya estén incluidas en `requirements.txt`, este paso se reduce a instalar las
dependencias del proyecto.)

### 2. Flujo completo con un nuevo CSV crudo

```bash
python -m src.train_pipeline \
  --input-path path\a\tus_pacientes.csv \
  --version-tag corrida_01
```

Esto ejecuta, en orden:

1. `ingest` → copia `tus_pacientes.csv` a `data/datasets/`.
2. `preprocess` → genera `X_train_raw`, `X_test_raw`, `y_train_raw`, `y_test_raw`.
3. `balance` → genera `X_train_balanced`, `y_train_balanced`.
4. `train` → entrena todos los modelos por defecto (`DEFAULT_MODELS`).
5. `evaluate` → calcula métricas en test.
6. `compare` → construye `data/results/model_comparison.csv`.

### 3. Reusar datasets ya preparados

Si ya tienes un raw en `data/datasets/` (por ejemplo generado desde tus
notebooks) y quieres saltarte la ingesta:

```bash
python -m src.train_pipeline \
  --steps preprocess,balance,train,evaluate,compare \
  --raw-path data/datasets/pacientes_raw_existente.csv
```

### 4. Re‑entrenar solo y reevaluar

```bash
python -m src.train_pipeline --steps train,evaluate,compare
```

### 5. Entrenar solo algunos modelos y con grid search

```bash
python -m src.train_pipeline \
  --steps train,evaluate,compare \
  --models logistic_regression,random_forest \
  --use-gridsearch
```

Los resultados de `GridSearchCV` se guardarán como:

- `data/results/logistic_regression/gridsearch_results_raw.csv`
- `data/results/random_forest/gridsearch_results_balanced.csv`, etc.

### 6. Generar predicciones para nuevos pacientes

Supón que ya tienes entrenado, por ejemplo, `random_forest` con SMOTE.

```bash
python -m src.predict \
  --input-path path\a\nuevos_pacientes.csv \
  --model-name random_forest \
  --dataset-type balanced
```

Esto genera un archivo en `data/results/predictions/` con las columnas
originales + `prediction` + probabilidades por clase.

---

Con este esquema, todo el trabajo que ya hiciste en los notebooks se
formaliza en pipelines reproducibles: solo falta que termines de rellenar la
función `basic_cleaning` (y, si quieres, algunas funciones de gráficos en
`evaluate.py`) copiando la lógica que ya sabes que funciona.
