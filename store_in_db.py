import chromadb
from chromadb.utils import embedding_functions

# 1. Set up the "Translator" (The Embedding Function)
# This model runs locally on your PC and is free.
model_name = "all-MiniLM-L6-v2"
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

# 2. Initialize the Database
# This will create a folder named 'my_db' in your project directory
client = chromadb.PersistentClient(path="./my_db")

# 3. Create a "Collection" (Think of it as a table in a database)
collection = client.get_or_create_collection(
    name="religious_knowledge", 
    embedding_function=embedding_func
)

def save_to_database(extracted_data):
    for i, entry in enumerate(extracted_data):
        # We give each chunk a unique ID
        chunk_id = f"id_{i}"
        
        collection.add(
            ids=[chunk_id],
            documents=[entry["content"]],
            metadatas=[{"source": entry["source"], "page": entry["page"]}]
        )
    print(f"Successfully stored {len(extracted_data)} chunks in the database!")

# --- This part connects to your work from Step 1 ---
# (In a real app, you'd import 'data' from your first script)
# For now, let's assume 'data' is the list you got from 'ingest_pdfs.py'
# save_to_database(data)