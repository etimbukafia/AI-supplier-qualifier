import pandas as pd
from pymongo import MongoClient
from pathlib import Path
from dotenv import load_dotenv
import os
from google.genai import Client

# Load environment variables
load_dotenv()

# === Configuration ===
MONGODB_URI = os.getenv(
    "MONGO_URI"
)
DB_NAME = "supply_chain_qms"

gemini_client = Client(api_key=os.getenv("GEMINI_API_KEY"))

def ingest_csv_to_mongo(data_dir: Path, filename: str, collection_name: str):
    file_path = data_dir / filename
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    # Read CSV into DataFrame
    df = pd.read_csv(file_path)
    records = df.to_dict(orient="records")

    # Drop existing documents in the target collection (optional)
    db[collection_name].drop()

    # Insert records
    if records:
        db[collection_name].insert_many(records)
        print(f"Inserted {len(records)} documents into '{collection_name}'.")
    else:
        print(f"No records found in {file_path}")


if __name__ == "__main__":
    # === Connect to MongoDB Atlas ===
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]

    # === File paths ===
    data_dir = Path("./data")
    files_and_collections = {
        "public_risk_indicators.csv": "external_risk_indicators",
    }

    # === Ingestion ===
    for filename, collection in files_and_collections.items():
        ingest_csv_to_mongo(data_dir, filename, collection)

    # === Close connection ===
    client.close()


