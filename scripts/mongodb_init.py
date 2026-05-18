"""Initialize MongoDB collections/indexes for PDG.

Usage (PowerShell):
  # Ensure .env exists with MONGODB_URI
  python scripts/mongodb_init.py
"""

from src.db.mongo_store import MongoStore


def main() -> None:
    store = MongoStore.from_env()
    if store is None:
        raise SystemExit("MongoDB is not enabled. Set MONGODB_URI in .env")

    if not store.ping():
        raise SystemExit("Cannot connect to MongoDB. Check MONGODB_URI / service status.")

    store.ensure_indexes()
    print("MongoDB indexes ensured.")


if __name__ == "__main__":
    main()
