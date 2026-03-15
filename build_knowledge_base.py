import os
import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions

# --- CONFIGURATION ---
PATH_TO_PDFS = r"C:\Users\Home\Downloads\My AI Project\knowledge_source"
DB_PATH = "./my_db"

# 1. Setup the "Brain" (The Embedding Model)
# This model translates English into math that the database understands.
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# 2. Setup the Database
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name="religious_knowledge", embedding_function=embedding_func)

def run_pipeline():
    print("🚀 Starting Ingestion...")
    count = 0

    for root, dirs, files in os.walk(PATH_TO_PDFS):
        for filename in files:
            if filename.endswith(".pdf"):
                file_path = os.path.join(root, filename)
                category = os.path.basename(root)
                
                try:
                    doc = fitz.open(file_path)
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        text = page.get_text().strip()
                        
                        if text:
                            # Save to Database
                            # We use a unique ID for every page
                            chunk_id = f"{filename}_pg_{page_num+1}"
                            
                            collection.add(
                                ids=[chunk_id],
                                documents=[text],
                                metadatas=[{"source": f"{category} > {filename}", "page": page_num + 1}]
                            )
                            count += 1
                    doc.close()
                    print(f"✅ Indexed: {filename}")
                except Exception as e:
                    print(f"❌ Error with {filename}: {e}")

    print(f"\n✨ Done! Stored {count} pages in your local database.")

if __name__ == "__main__":
    run_pipeline()