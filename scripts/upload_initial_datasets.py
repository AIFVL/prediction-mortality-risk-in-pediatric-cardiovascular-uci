"""
Utility script to upload local datasets to MongoDB GridFS.
Run this locally before deploying to ensure the cloud backend can find the data.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.db.mongo_store import MongoStore
from src.config import DATASETS_DIR

def main():
    store = MongoStore.from_env()
    if not store:
        print("Error: MONGODB_URI not set in .env")
        return

    if not store.ping():
        print("Error: Could not connect to MongoDB")
        return

    datasets_to_upload = [
        "dataset_inicial.xlsx",
        "dataset_limpio.xlsx",
    ]

    print(f"Checking for datasets in {DATASETS_DIR}...")
    
    for filename in datasets_to_upload:
        path = DATASETS_DIR / filename
        if path.exists():
            print(f"Uploading {filename}...")
            try:
                file_id = store.upload_path(
                    path, 
                    metadata={"kind": "initial_dataset", "uploaded_locally": True}
                )
                print(f"Successfully uploaded {filename} (ID: {file_id})")
            except Exception as e:
                print(f"Failed to upload {filename}: {e}")
        else:
            print(f"Warning: {filename} not found locally at {path}")

    print("\nUpload process finished.")

if __name__ == "__main__":
    main()
