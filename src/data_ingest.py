from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd

from src.config import DATASETS_DIR, MONGODB_URI
from src.io_utils import ensure_dir, write_csv, file_hash, save_json, build_versioned_name

try:
    from src.db.mongo_store import MongoStore
except ImportError:
    MongoStore = None


def run_ingest(input_path: str, version_tag: Optional[str] = None) -> Path:
    input_file = Path(input_path)
    
    # Si el archivo no existe localmente, intentamos descargarlo de MongoDB
    if not input_file.exists() and MONGODB_URI:
        print(f"File {input_file} not found locally. Searching in MongoDB...")
        store = MongoStore.from_env()
        if store:
            success = store.download_file_by_name(input_file.name, input_file)
            if success:
                print(f"Successfully downloaded {input_file.name} from MongoDB.")
            else:
                print(f"Could not find {input_file.name} in MongoDB.")

    if not input_file.exists():
        raise FileNotFoundError(f"Input data file not found: {input_file}")

    if input_file.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = build_versioned_name("pacientes_raw", version_tag or timestamp)

    ensure_dir(DATASETS_DIR)
    output_path = DATASETS_DIR / f"{base_name}.csv"
    write_csv(df, output_path)

    metadata = {
        "source_path": str(input_file),
        "stored_path": str(output_path),
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "hash_sha256": file_hash(output_path),
        "ingest_timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    meta_path = DATASETS_DIR / f"{base_name}_metadata.json"
    save_json(metadata, meta_path)

    return output_path
