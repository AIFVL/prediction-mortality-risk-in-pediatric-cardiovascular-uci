from pathlib import Path
from typing import Optional, Tuple
import datetime
import unicodedata

import numpy as np
import pandas as pd
import warnings

from src.config import PROCESSED_DIR, TARGET_COLUMN, RANDOM_STATE, PREPROCESS_MODE, MONGODB_URI
from src.io_utils import ensure_dir, read_csv, write_csv, save_json

try:
    from src.db.mongo_store import MongoStore
except ImportError:
    MongoStore = None

DATASET_LIMPIO_PATH = Path(__file__).resolve().parents[1] / "data" / "datasets" / "dataset_limpio.xlsx"

DROP_COLS = [
    "EPISODIO", "Nº documento", "NOMBRE",
    "Descripción Aseguradora", "Clase de Aseguradora",
    "MORTALIDAD A 30 DÍAS",
]

INFERENCE_DROP_COLS = DROP_COLS + [
    TARGET_COLUMN,
    "MORTALIDAD GENERAL",
    "MORTALIDAD A 30 DÍAS",
    "Mortalidad",
]

MORTALIDAD_MULTICLASS_MAP = {
    "no murio": 0,
    "murió en los primeros 30 dias": 1,
    "murió despues de 30 dias": 2,
    "murio": 1,
    "murio en los primeros 30 dias": 1,
    "murio despues de 30 dias": 2,
    "murió (>30d)": 2,
    "murio (>30d)": 2,
    "murió (<30d)": 1,
    "murio (<30d)": 1,
}

MULTICLASS_TO_BINARY_MAP = {0: 0, 1: 1, 2: 1}

MORTALITY_BINARY_MAP = {
    "no murio": 0,
    "no murió": 0,
    "murio": 1,
    "murió": 1,
    "murio (>30d)": 1,
    "murió (>30d)": 1,
    "murio (<30d)": 1,
    "murió (<30d)": 1,
    "murió en los primeros 30 dias": 1,
    "murio en los primeros 30 dias": 1,
    "murió despues de 30 dias": 1,
    "murio despues de 30 dias": 1,
}

NORMALIZED_MORTALITY_BINARY_MAP = {
    "no murio": 0,
    "murio": 1,
    "murio (>30d)": 1,
    "murio (<30d)": 1,
    "murio en los primeros 30 dias": 1,
    "murio despues de 30 dias": 1,
}


NOTEBOOK_NULL_AS_NO_APLICA = [
    "Disfunción Multiorganica",
    "Arritmias",
    "Sangrado",
    "ISO",
    "Sepsis",
    "Fecha Fin ECMO",
    "Fecha Inicio ECMO",
    "Tipo Ventilación Mecanica",
    "Medicamentos-Antibiotico Profilaxis",
    "Antibióticos Profilácticos",
]

NOTEBOOK_TIME_TO_HOURS_COLS = [
    "Paro Circulatorio Hora Total",
    "Bomba Hora Total",
    "Isquemia Hora Total",
    "Tiempo quirurgico ",
    "TIEMPO QUIROFANO",
]

NOTEBOOK_ZERO_FILL_COLS = [
    "Paro Circulatorio Hora Total",
    "Isquemia Hora Total",
    "Bomba Hora Total",
    "Días ECMO",
]

NOTEBOOK_DOSIS_ZERO_COLS = [
    "Dosis Inicial Cardioplejia",
    "Dosis Mantenimiento Cardioplejia",
    "Volumen total cardioplejia ",
]


def _normalize_label(value) -> str:
    text = str(value).strip().lower()
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _derive_target(df: pd.DataFrame) -> pd.Series:
    """Create the binary Mortalidad target: 0 = no murió, 1 = murió."""
    df_temp = df.copy()
    df_temp.columns = df_temp.columns.str.strip()

    mort_general = df_temp.get("MORTALIDAD GENERAL", None)
    if mort_general is None:
        raise ValueError("Column 'MORTALIDAD GENERAL' not found.")

    si_general = mort_general.astype(str).str.strip().str.upper() == "SI"
    target = si_general.astype(int)
    target.name = TARGET_COLUMN
    return target


def _drop_rows_with_many_nulls(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    min_non_null = int(df.shape[1] * (1 - threshold))
    non_null_counts = df.count(axis=1)
    return df.loc[non_null_counts >= min_non_null].copy()


def _convert_to_hours(value) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, datetime.time):
        return value.hour + value.minute / 60 + value.second / 3600
    if isinstance(value, datetime.datetime):
        return value.hour + value.minute / 60 + value.second / 3600
    if isinstance(value, (int, float, np.number)):
        return float(value)
    try:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return np.nan
        return parsed.hour + parsed.minute / 60 + parsed.second / 3600
    except Exception:
        return np.nan


def _unify_age_years(edad_val, unidad):
    if pd.isna(edad_val) or pd.isna(unidad):
        return np.nan
    try:
        edad_num = float(edad_val)
    except (ValueError, TypeError):
        return np.nan
    unidad_clean = str(unidad).upper().strip()
    if unidad_clean in ["AÑOS", "AÑO", "A"]:
        return edad_num
    if unidad_clean in ["MESES", "M"]:
        return edad_num / 12
    if unidad_clean in ["DIAS", "D"]:
        return edad_num / 365.25
    return np.nan


def _combine_mortalidad(row) -> Optional[str]:
    mg = str(row.get("MORTALIDAD GENERAL", "")).strip().upper()
    m30 = str(row.get("MORTALIDAD A 30 DÍAS", "")).strip().upper()
    if mg == "NO" and m30 == "NO":
        return "no murio"
    if mg == "SI" and m30 == "SI":
        return "murió en los primeros 30 dias"
    if mg == "SI" and m30 == "NO":
        return "murió despues de 30 dias"
    return None


def _notebook_v1_prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    row_counts: dict[str, int] = {}

    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.strip()

    row_counts["start"] = len(df_clean)

    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    row_counts["drop_duplicates"] = len(df_clean)

    df_clean = _drop_rows_with_many_nulls(df_clean, threshold=0.5)
    row_counts["drop_rows_50pct_null"] = len(df_clean)

    for col in NOTEBOOK_NULL_AS_NO_APLICA:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna("NO APLICA")

    for col in NOTEBOOK_TIME_TO_HOURS_COLS:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(_convert_to_hours)

    if "EDAD" in df_clean.columns and "D - M - A" in df_clean.columns:
        df_clean["EDAD_AÑOS"] = df_clean.apply(
            lambda row: _unify_age_years(row["EDAD"], row["D - M - A"]), axis=1
        )
        df_clean = df_clean.drop(columns=["EDAD", "D - M - A"])

    if "EDAD_AÑOS" in df_clean.columns:
        df_clean = df_clean[(df_clean["EDAD_AÑOS"] < 18) | (df_clean["EDAD_AÑOS"].isna())].copy()
        row_counts["filter_pediatric_or_na"] = len(df_clean)
        df_clean = df_clean[df_clean["EDAD_AÑOS"].notna()].copy()
        row_counts["drop_missing_edad"] = len(df_clean)

    missing_percent = 100 * df_clean.isnull().sum() / len(df_clean)
    columns_to_drop = missing_percent[missing_percent >= 75].index
    df_clean = df_clean.drop(columns=columns_to_drop)

    if "Peso" in df_clean.columns:
        df_clean["Peso"] = pd.to_numeric(df_clean["Peso"], errors="coerce")
        mask_gramos = (df_clean["Peso"] > 200) & df_clean["Peso"].notna()
        df_clean.loc[mask_gramos, "Peso"] = df_clean.loc[mask_gramos, "Peso"] / 1000

    if "Talla" in df_clean.columns:
        df_clean["Talla"] = pd.to_numeric(df_clean["Talla"], errors="coerce")
        mask_talla_valida = (df_clean["Talla"].between(15, 250)) | df_clean["Talla"].isna()
        df_clean = df_clean[mask_talla_valida].copy()
        row_counts["filter_talla"] = len(df_clean)

    if "IMC" in df_clean.columns:
        df_clean["IMC"] = pd.to_numeric(df_clean["IMC"], errors="coerce")
        df_clean.loc[df_clean["IMC"] > 100, "IMC"] = np.nan

    if {"Talla", "IMC", "Peso"}.issubset(df_clean.columns):
        mask_talla_faltante = df_clean["Talla"].isna()
        mask_tiene_imc_peso = df_clean["IMC"].notna() & df_clean["Peso"].notna()
        mask_calcular_talla = mask_talla_faltante & mask_tiene_imc_peso
        if mask_calcular_talla.any():
            df_clean.loc[mask_calcular_talla, "Talla"] = 100 * np.sqrt(
                df_clean.loc[mask_calcular_talla, "Peso"]
                / df_clean.loc[mask_calcular_talla, "IMC"]
            )

    if "Dosis Inicial Cardioplejia" in df_clean.columns and "Talla" in df_clean.columns:
        mask_entrenamiento = df_clean["Talla"].notna() & df_clean["Dosis Inicial Cardioplejia"].notna()
        if mask_entrenamiento.sum() > 10:
            x = df_clean.loc[mask_entrenamiento, "Dosis Inicial Cardioplejia"].astype(float)
            y = df_clean.loc[mask_entrenamiento, "Talla"].astype(float)
            corr = x.corr(y)
            if pd.notna(corr) and abs(corr) >= 0.3:
                try:
                    slope, intercept = np.polyfit(x, y, 1)
                    mask_imputar = df_clean["Talla"].isna() & df_clean["Dosis Inicial Cardioplejia"].notna()
                    if mask_imputar.any():
                        x_imp = df_clean.loc[mask_imputar, "Dosis Inicial Cardioplejia"].astype(float)
                        talla_imputada = (slope * x_imp + intercept).clip(30, 200)
                        df_clean.loc[mask_imputar, "Talla"] = talla_imputada
                except Exception:
                    pass

    if {"Talla", "IMC", "Peso"}.issubset(df_clean.columns):
        mask_imc_faltante = df_clean["IMC"].isna()
        mask_tiene_peso_talla = df_clean["Peso"].notna() & df_clean["Talla"].notna()
        mask_calcular_imc = mask_imc_faltante & mask_tiene_peso_talla
        if mask_calcular_imc.any():
            df_clean.loc[mask_calcular_imc, "IMC"] = (
                df_clean.loc[mask_calcular_imc, "Peso"]
                / ((df_clean.loc[mask_calcular_imc, "Talla"] / 100) ** 2)
            )

    if "IMC" in df_clean.columns:
        df_clean = df_clean[df_clean["IMC"].notna()].copy()
        row_counts["drop_missing_imc"] = len(df_clean)

    for col in NOTEBOOK_ZERO_FILL_COLS + NOTEBOOK_DOSIS_ZERO_COLS:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0)

    if "FECHA CX" in df_clean.columns:
        df_clean = df_clean.dropna(subset=["FECHA CX"]).copy()
        row_counts["drop_missing_fecha_cx"] = len(df_clean)

    if "ASA" in df_clean.columns:
        df_clean["ASA"] = df_clean["ASA"].fillna("III")

    mortalidad = df_clean.apply(_combine_mortalidad, axis=1)
    mortalidad = mortalidad.dropna()
    df_clean = df_clean.loc[mortalidad.index].copy()
    row_counts["drop_missing_mortalidad"] = len(df_clean)

    y_encoded = mortalidad.map(_normalize_label).map(NORMALIZED_MORTALITY_BINARY_MAP)
    unknown_targets = sorted(mortalidad[y_encoded.isna()].dropna().astype(str).unique())
    if unknown_targets:
        raise ValueError(f"Unknown Mortalidad labels: {unknown_targets}")
    y_encoded = y_encoded.astype(int)
    y_encoded.name = TARGET_COLUMN

    df_clean = df_clean.drop(columns=[c for c in ["MORTALIDAD GENERAL", "MORTALIDAD A 30 DÍAS"] if c in df_clean.columns])
    df_clean = df_clean.drop(columns=[c for c in DROP_COLS if c in df_clean.columns], errors="ignore")

    return df_clean, y_encoded, row_counts


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.strip()

    df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    for col in ["Peso", "Talla", "IMC"]:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    if {"Talla", "Peso", "IMC"}.issubset(df_clean.columns):
        mask_talla_faltante = df_clean["Talla"].isna()
        mask_tiene_imc_peso = df_clean["IMC"].notna() & df_clean["Peso"].notna()
        mask_calcular_talla = mask_talla_faltante & mask_tiene_imc_peso
        if mask_calcular_talla.any():
            df_clean.loc[mask_calcular_talla, "Talla"] = 100 * np.sqrt(
                df_clean.loc[mask_calcular_talla, "Peso"]
                / df_clean.loc[mask_calcular_talla, "IMC"]
            )

    if "IMC" in df_clean.columns:
        df_clean.loc[df_clean["IMC"] > 100, "IMC"] = np.nan

    if {"IMC", "Peso", "Talla"}.issubset(df_clean.columns):
        mask_imc_faltante = df_clean["IMC"].isna()
        mask_tiene_peso_talla = df_clean["Peso"].notna() & df_clean["Talla"].notna()
        mask_calcular_imc = mask_imc_faltante & mask_tiene_peso_talla
        if mask_calcular_imc.any():
            talla_m = df_clean.loc[mask_calcular_imc, "Talla"] / 100.0
            df_clean.loc[mask_calcular_imc, "IMC"] = (
                df_clean.loc[mask_calcular_imc, "Peso"] / (talla_m ** 2)
            )

    if "IMC" in df_clean.columns:
        df_clean = df_clean[df_clean["IMC"].notna()].copy()

    for col in ["Paro Circulatorio Hora Total", "Isquemia Hora Total"]:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0)

    return df_clean


def _basic_inference_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.strip()

    for col in ["Peso", "Talla", "IMC"]:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    if {"Talla", "Peso", "IMC"}.issubset(df_clean.columns):
        mask_talla_faltante = df_clean["Talla"].isna()
        mask_tiene_imc_peso = df_clean["IMC"].notna() & df_clean["Peso"].notna()
        mask_calcular_talla = mask_talla_faltante & mask_tiene_imc_peso
        if mask_calcular_talla.any():
            df_clean.loc[mask_calcular_talla, "Talla"] = 100 * np.sqrt(
                df_clean.loc[mask_calcular_talla, "Peso"]
                / df_clean.loc[mask_calcular_talla, "IMC"]
            )

    if "IMC" in df_clean.columns:
        df_clean.loc[df_clean["IMC"] > 100, "IMC"] = np.nan

    if {"IMC", "Peso", "Talla"}.issubset(df_clean.columns):
        mask_imc_faltante = df_clean["IMC"].isna()
        mask_tiene_peso_talla = df_clean["Peso"].notna() & df_clean["Talla"].notna()
        mask_calcular_imc = mask_imc_faltante & mask_tiene_peso_talla
        if mask_calcular_imc.any():
            talla_m = df_clean.loc[mask_calcular_imc, "Talla"] / 100.0
            df_clean.loc[mask_calcular_imc, "IMC"] = (
                df_clean.loc[mask_calcular_imc, "Peso"] / (talla_m ** 2)
            )

    for col in ["Paro Circulatorio Hora Total", "Isquemia Hora Total"]:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0)

    return df_clean


def _ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def _safe_label_transform(series: pd.Series, encoder) -> pd.Series:
    classes = list(getattr(encoder, "classes_", []))
    mapping = {cls: idx for idx, cls in enumerate(classes)}
    return series.astype(str).map(mapping).fillna(-1).astype(int)


def preprocess_for_inference(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same notebook-2 preprocessing artifacts saved during training.
    This is robust to missing columns and incomplete inputs.
    """
    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    df = df.drop(columns=[c for c in INFERENCE_DROP_COLS if c in df.columns], errors="ignore")
    df = _basic_inference_cleaning(df)

    preprocessor_path = PROCESSED_DIR / "preprocessor.pkl"
    preprocessor = None
    if preprocessor_path.exists():
        import pickle as _pickle
        with preprocessor_path.open("rb") as _f:
            preprocessor = _pickle.load(_f)

    if preprocessor:
        date_cols_to_drop = preprocessor.get("date_cols_to_drop", [])
        df = df.drop(columns=[c for c in date_cols_to_drop if c in df.columns], errors="ignore")

        le_asa = preprocessor.get("le_asa")
        if le_asa is not None and "ASA" in df.columns:
            df["ASA_encoded"] = _safe_label_transform(df["ASA"], le_asa)
            df = df.drop(columns=["ASA"])

        nominales_presentes = preprocessor.get("nominales_presentes", [])
        ohe_col_names = preprocessor.get("ohe_col_names", [])
        if nominales_presentes:
            nominales_presentes = [c for c in nominales_presentes if c in df.columns]
        if nominales_presentes:
            df_ohe = pd.get_dummies(df[nominales_presentes], prefix="cat", drop_first=False)
            df = df.drop(columns=nominales_presentes)
            df = pd.concat([df, df_ohe], axis=1)

        label_encoders = preprocessor.get("label_encoders", {})
        for col, le in label_encoders.items():
            if col in df.columns:
                df[col] = _safe_label_transform(df[col], le)

        median_values = preprocessor.get("median_values")
        if isinstance(median_values, pd.Series):
            df = df.fillna(median_values)
        df = df.fillna(0)

        for col in df.select_dtypes(include=["bool"]).columns:
            df[col] = df[col].astype(int)

        df = _ensure_unique_columns(df)

        numeric_cols = preprocessor.get("numeric_cols", [])
        scaler = preprocessor.get("scaler")
        if scaler is not None and numeric_cols:
            for col in numeric_cols:
                if col not in df.columns:
                    df[col] = 0
            df[numeric_cols] = scaler.transform(df[numeric_cols])

        if ohe_col_names:
            for col in ohe_col_names:
                if col not in df.columns:
                    df[col] = 0

    df = _ensure_unique_columns(df)

    X_train_path = PROCESSED_DIR / "csv" / "X_train_raw.csv"
    if X_train_path.exists():
        train_cols = read_csv(X_train_path).columns
        for col in train_cols:
            if col not in df.columns:
                df[col] = 0
        df = df.reindex(columns=train_cols, fill_value=0)
    else:
        df = encode_features(df)

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.preprocessing import LabelEncoder
    df_enc = df.copy()

    date_cols = []
    for col in df_enc.columns:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(df_enc[col], errors="coerce")
            if parsed.notna().sum() > len(df_enc) * 0.5:
                date_cols.append(col)
        except Exception:
            pass
    for col in date_cols:
        if col in df_enc.columns:
            df_enc.drop(columns=[col], inplace=True)

    cat_cols = df_enc.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str).str.strip())

    medians = df_enc.median(numeric_only=True)
    df_enc = df_enc.fillna(medians)
    df_enc = df_enc.fillna(0)

    return df_enc


NOTEBOOK2_DATE_COLS = [
    "Hora Inicio Cardioplejia",
    "Fecha de Admisión",
    "Fecha Ingreso UCIPCV",
    "Fecha Egreso Vivo",
    "FECHA CX",
    "Fecha Junta Medica",
    "Fecha Intubación",
    "Fecha Extubación",
]

NOTEBOOK2_DROP_COLS = [
    "Días desde la Junta a la Cx",
    "Estancia UCIPCV",
    "Días des de ingreso hasta UCI",
    "Días desde ingreso a cx",
]

NOTEBOOK2_VARIABLES_NOMINALES = [
    "SEXO",
    "Prematuridad",
    "PROCEDIMIENTO",
    "RACHS-1",
    "CEC",
    "CLAMP",
    "ARRESTO CIRCULATORIO",
    "Uso de OCTAPLEX u OCTAPLAS",
    "Antibióticos Profilácticos",
    "Medicamentos-Antibiotico Profilaxis",
    "Uso de Mupirocina",
    "Ventilación Mecanica",
    "Tipo Ventilación Mecanica",
    "ECMO",
    "Reintervenciones asociadas al ECMO",
    "Reintervención Qx",
    "Número de reintervenciones",
    "ISO",
    "Arritmias",
    "Sepsis",
    "Disfunción Multiorganica",
    "Sangrado",
    "Uso de Hemoderivados en el Episodio",
    "Valoración por nutrición",
    "Junta Medica",
]


def _load_from_dataset_limpio() -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Load and process dataset_limpio.xlsx exactly as notebook 2 does:
      1. Drop date and duration columns (notebook 2, cell 9)
      2. LabelEncode ASA (ordinal)
      3. One-Hot Encode nominal categorical variables
      4. StandardScaler on numeric columns
      5. Map Mortalidad multiclass → binary (0=survived, 1=died)
    """
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    import openpyxl  # noqa: F401 – ensure openpyxl is available
    df = pd.read_excel(DATASET_LIMPIO_PATH)
    df.columns = df.columns.str.strip()

    row_counts: dict[str, int] = {"start": len(df)}

    # --- Drop date and duration columns (notebook 2, cell 9) ---
    cols_to_drop = [c for c in NOTEBOOK2_DATE_COLS + NOTEBOOK2_DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # --- Locate and extract the target column ---
    target_col = None
    for col in df.columns:
        if "Mortalidad" in col:
            target_col = col
            break
    if target_col is None:
        raise ValueError("Column 'Mortalidad' not found in dataset_limpio.xlsx")

    # Map exactly as notebook 2: multiclass string → int → binary int
    y_multiclass = (
        df[target_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(MORTALIDAD_MULTICLASS_MAP)
    )
    unknown = df[target_col][y_multiclass.isna()].dropna().unique()
    if len(unknown):
        raise ValueError(f"Unknown Mortalidad labels in dataset_limpio.xlsx: {unknown.tolist()}")

    y_binary = y_multiclass.map(MULTICLASS_TO_BINARY_MAP).astype(int)
    y_binary.name = TARGET_COLUMN
    row_counts["after_target_mapping"] = int(y_binary.notna().sum())

    X = df.drop(columns=[target_col])

    # --- ASA: LabelEncoding (ordinal), notebook 2 cell 9 ---
    if "ASA" in X.columns:
        le_asa = LabelEncoder()
        X["ASA_encoded"] = le_asa.fit_transform(X["ASA"].astype(str))
        X = X.drop(columns=["ASA"])

    # --- OHE for nominal categorical variables (notebook 2, cell 9) ---
    # Keep as bool dtype intentionally: notebook 2's pd.get_dummies produces bool,
    # which pandas excludes from select_dtypes(include=[np.number]), so these columns
    # are NOT passed to StandardScaler — exactly replicating notebook 2's behavior.
    nominales_presentes = [c for c in NOTEBOOK2_VARIABLES_NOMINALES if c in X.columns]
    ohe_cols: list[str] = []
    if nominales_presentes:
        X_ohe = pd.get_dummies(X[nominales_presentes], prefix="cat", drop_first=False)
        ohe_cols = X_ohe.columns.tolist()
        X = X.drop(columns=nominales_presentes)
        X = pd.concat([X, X_ohe], axis=1)

    # --- LabelEncode any remaining object columns ---
    label_encoders: dict = {}
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # --- Fill NaNs with column median then 0 ---
    median_values = X.median(numeric_only=True)
    X = X.fillna(median_values).fillna(0)

    # --- StandardScaler on numeric columns, excluding OHE bool columns
    # (notebook 2 has bool OHE cols which are skipped by select_dtypes(np.number)) ---
    numeric_cols = [
        c for c in X.select_dtypes(include=[np.number]).columns
        if c not in ohe_cols
    ]
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # --- Convert OHE bool columns to int so ADASYN can compute differences ---
    for col in X.select_dtypes(include=["bool"]).columns:
        X[col] = X[col].astype(int)

    # --- Save preprocessor artifacts so inference can replicate these transforms ---
    import pickle as _pickle
    _preprocessor = {
        "scaler": scaler,
        "numeric_cols": numeric_cols,
        "nominales_presentes": nominales_presentes,
        "ohe_col_names": ohe_cols,
        "le_asa": le_asa if "ASA" in df.columns else None,
        "label_encoders": label_encoders,
        "median_values": median_values,
        "date_cols_to_drop": NOTEBOOK2_DATE_COLS + NOTEBOOK2_DROP_COLS,
    }
    ensure_dir(PROCESSED_DIR)
    with open(PROCESSED_DIR / "preprocessor.pkl", "wb") as _f:
        _pickle.dump(_preprocessor, _f)

    return X, y_binary, row_counts


def run_preprocessing(
    raw_path: str,
    version_tag: Optional[str] = None,
    mode: str = PREPROCESS_MODE,
) -> Tuple[Path, Path]:
    raw_file = Path(raw_path)

    row_counts = {}
    already_encoded = False

    # Si el archivo no existe localmente, intentamos descargarlo de MongoDB
    if not DATASET_LIMPIO_PATH.exists() and MONGODB_URI:
        print(f"File {DATASET_LIMPIO_PATH} not found locally. Searching in MongoDB...")
        store = MongoStore.from_env()
        if store:
            success = store.download_file_by_name(DATASET_LIMPIO_PATH.name, DATASET_LIMPIO_PATH)
            if success:
                print(f"Successfully downloaded {DATASET_LIMPIO_PATH.name} from MongoDB.")
            else:
                print(f"Could not find {DATASET_LIMPIO_PATH.name} in MongoDB.")

    # --- Priority: use dataset_limpio.xlsx when available (notebook workflow) ---
    if DATASET_LIMPIO_PATH.exists():
        df_clean, y_encoded, row_counts = _load_from_dataset_limpio()
        already_encoded = True  # OHE + StandardScaler already applied inside
    elif not raw_file.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_file}")
    elif mode == "notebook_v1":
        if not raw_file.exists():
            raise FileNotFoundError(f"Raw dataset not found: {raw_file}")
        df = read_csv(raw_file)
        df.columns = df.columns.str.strip()
        df_clean, y_encoded, row_counts = _notebook_v1_prepare(df)
    else:
        if not raw_file.exists():
            raise FileNotFoundError(f"Raw dataset not found: {raw_file}")
        df = read_csv(raw_file)
        df.columns = df.columns.str.strip()
        if TARGET_COLUMN in df.columns:
            y_raw_str = df[TARGET_COLUMN].copy()
            y_encoded = y_raw_str.map(_normalize_label).map(NORMALIZED_MORTALITY_BINARY_MAP)
            unknown_targets = sorted(y_raw_str[y_encoded.isna()].dropna().astype(str).unique())
            if unknown_targets:
                raise ValueError(f"Unknown Mortalidad labels: {unknown_targets}")
            y_encoded = y_encoded.astype(int)
            y_encoded.name = TARGET_COLUMN
            df_clean = basic_cleaning(df.drop(columns=[TARGET_COLUMN]))
            y_encoded = y_encoded.loc[df_clean.index]
        else:
            y_raw = _derive_target(df)
            cols_to_drop = [c for c in DROP_COLS if c in df.columns]
            df_tmp = df.drop(columns=cols_to_drop, errors="ignore")
            df_clean = basic_cleaning(df_tmp)
            y_encoded = y_raw.loc[df_clean.index].astype(int).copy()
            y_encoded.name = TARGET_COLUMN

    save_json({"classes": ["no murio", "murio"]}, PROCESSED_DIR / "target_classes.json")

    X = df_clean if already_encoded else encode_features(df_clean)

    ensure_dir(PROCESSED_DIR / "csv")

    X_train_path = PROCESSED_DIR / "csv" / "X_train_raw.csv"
    y_train_path = PROCESSED_DIR / "csv" / "y_train_raw.csv"

    write_csv(X, X_train_path)
    write_csv(y_encoded.to_frame(name=TARGET_COLUMN), y_train_path)

    tag = version_tag or raw_file.stem
    meta = {
        "source_raw": str(raw_file),
        "X_train_shape": list(X.shape),
        "y_train_shape": [int(y_encoded.shape[0])],
        "variable_objetivo": TARGET_COLUMN,
        "random_state": RANDOM_STATE,
        "version_tag": tag,
        "target_classes": ["no murio", "murio"],
        "class_counts": {str(k): int(v) for k, v in y_encoded.value_counts().sort_index().items()},
        "preprocess_mode": mode,
        "row_counts": {k: int(v) for k, v in row_counts.items()},
    }

    save_json(meta, PROCESSED_DIR / "metadatos_generated.json")

    return X_train_path, y_train_path
