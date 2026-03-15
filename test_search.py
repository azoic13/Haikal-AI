import chromadb
from chromadb.utils import embedding_functions

# 1. CONNECT TO YOUR BRAIN
client = chromadb.PersistentClient(path="./my_db")
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
collection = client.get_collection(
    name="religious_knowledge", 
    embedding_function=embedding_func
)

def ask_the_brain(query):
    # 2. SEARCH FOR THE TOP 2 RESULTS
    results = collection.query(
        query_texts=[query],
        n_results=2
    )

    print(f"\n🔎 Searching for: '{query}'")
    print("-" * 50)

    for i in range(len(results['documents'][0])):
        content = results['documents'][0][i]
        source = results['metadatas'][0][i]['source']
        page = results['metadatas'][0][i]['page']
        
        print(f"📍 [RESULT {i+1}]")
        print(f"📖 Source: {source} (Page {page})")
        print(f"📝 Text: {content[:300]}...") # Shows only the first 300 characters
        print("-" * 50)

# 3. TRY IT OUT
# You can type a question here in English or Arabic!
user_query = input("Ask a question (English or Arabic): ")
ask_the_brain(user_query)