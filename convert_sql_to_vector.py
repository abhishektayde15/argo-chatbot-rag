import pandas as pd
from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import os

# --- SQL से वेक्टर DB में डेटा डालना (SQL to Vector DB) ---
def process_sql_to_vector_db(db_file):
    """Reads data from SQL and stores embeddings in ChromaDB."""
    print(f"✅ Processing SQL data from '{db_file}' for vectorization...")

    engine = create_engine(f"sqlite:///{db_file}")

    print("⏳ Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model loaded.")

    # ✅ यहाँ PersistentClient का इस्तेमाल करो
    client = PersistentClient(path="./chroma")
    collection = client.get_or_create_collection("argo_summaries")

    batch_size = 1000

    total_records_df = pd.read_sql("SELECT COUNT(*) FROM argo_profiles", engine)
    total_records = total_records_df.iloc[0, 0]

    def generate_summary(row):
        return (
            f"Argo float {row['float_id']} profile taken on {row['time'].strftime('%Y-%m-%d')} "
            f"at coordinates ({row['lat']:.2f}, {row['lon']:.2f}). "
            f"Data includes temperature of {row['temperature']:.2f}°C, salinity of {row['salinity']:.2f} PSU, and pressure at {row['depth']:.2f} dbar."
        )

    for offset in range(0, total_records, batch_size):
        query = f"SELECT * FROM argo_profiles LIMIT {batch_size} OFFSET {offset}"
        df_batch = pd.read_sql(query, engine)
        
        df_batch['time'] = pd.to_datetime(df_batch['time'])
        
        summaries = [generate_summary(row) for _, row in df_batch.iterrows()]
        
        embeddings = model.encode(summaries).tolist()
        
        metadatas = df_batch[['float_id', 'profile_number', 'time', 'lat', 'lon']].astype(str).to_dict('records')
        ids = [f"profile_{row['float_id']}_{row['profile_number']}_{idx}" for idx, row in df_batch.iterrows()]
        
        collection.add(
            embeddings=embeddings,
            documents=summaries,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Successfully inserted {len(df_batch)} embeddings into ChromaDB. Offset: {offset}")

# --- Main Execution Block ---
if __name__ == "__main__":
    database_file_path = "argo.db"

    print("\n⏳ Deleting existing ChromaDB collection to prevent duplicates...")
    try:
        client = PersistentClient(path="./chroma")
        client.delete_collection("argo_summaries")
        print("✅ Collection 'argo_summaries' deleted.")
    except Exception as e:
        print(f"⚠️ Could not delete collection. It may not exist. {e}")
    
    process_sql_to_vector_db(database_file_path)
    
    print("\n🎉 All data processing complete!")
    print("Now you can proceed to the next steps of building the RAG pipeline.")
