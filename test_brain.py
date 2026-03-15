import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./my_db")
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
collection = client.get_collection(name="religious_knowledge", embedding_function=embedding_func)

def ask(query):
    results = collection.query(query_texts=[query], n_results=1)
    print(f"\nQuery: {query}")
    print(f"Top Source: {results['metadatas'][0][0]['source']}")
    print(f"Content: {results['documents'][0][0][:200]}...")

# Test with both!
ask("How to perform prayer?")
ask("كيفية أداء الصلاة؟")