import streamlit as st
import os
import sys

# --- STEP 1: SQLITE FIX ---
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass 

import chromadb
from chromadb.utils import embedding_functions
from youtubesearchpython import VideosSearch
from youtube_transcript_api import YouTubeTranscriptApi
from google import genai
from google.genai import types
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 2. INITIAL SETUP ---
st.set_page_config(page_title="Haikal AI", page_icon="🕌", layout="wide")

if "GEMINI_API_KEY" not in st.secrets:
    st.error("Please add 'GEMINI_API_KEY' to Streamlit Secrets.")
    st.stop()

client_gemini = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_sources" not in st.session_state:
    st.session_state.current_sources = {"pdfs": [], "vids": []}

# --- 3. DATABASE (REPAIRED INGESTION) ---
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

@st.cache_resource
def load_and_ingest_db():
    # Use a specific path for Streamlit persistence
    persist_directory = "./chroma_db"
    client_db = chromadb.PersistentClient(path=persist_directory)
    collection = client_db.get_or_create_collection(
        name="religious_knowledge", 
        embedding_function=embedding_func
    )

    # CHECK IF FILES EXIST
    knowledge_path = "./knowledge"
    if not os.path.exists(knowledge_path):
        st.error(f"❌ Folder '{knowledge_path}' not found in GitHub!")
        return collection

    pdf_files = [f for f in os.listdir(knowledge_path) if f.endswith('.pdf')]
    st.sidebar.info(f"📁 Found {len(pdf_files)} PDFs in folder.")

    if collection.count() == 0 and len(pdf_files) > 0:
        with st.status("🛠️ Indexing library (First time only)..."):
            # Try loading without recursive first if it fails
            loader = PyPDFDirectoryLoader(knowledge_path)
            docs = loader.load()
            
            if docs:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800, 
                    chunk_overlap=100,
                    separators=["\n\n", "\n", "؟", ".", "،", " ", ""]
                )
                chunks = splitter.split_documents(docs)
                
                batch_size = 50 # Smaller batches for 1GB library
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i : i + batch_size]
                    collection.add(
                        ids=[f"id_{j}" for j in range(i, i + len(batch))],
                        documents=[chunk.page_content for chunk in batch],
                        metadatas=[chunk.metadata for chunk in batch]
                    )
                st.success(f"✅ Indexed {len(chunks)} text snippets!")
            else:
                st.warning("⚠️ Files found, but LangChain could not read them.")
    
    return collection

collection = load_and_ingest_db()

# --- 4. CORE UTILITIES ---

def get_data(query, search_mode):
    context, pdf_sources, yt_sources = "", [], []
    
    if search_mode in ["Search Local Books Only", "Hybrid (Both)"]:
        try:
            # INCREASE n_results to 7 for 1GB library
            res = collection.query(query_texts=[query], n_results=7)
            if res['documents'] and len(res['documents'][0]) > 0:
                for d, m in zip(res['documents'][0], res['metadatas'][0]):
                    s_name = os.path.basename(m.get('source', 'Unknown Book'))
                    pdf_sources.append(s_name)
                    context += f"\n[SOURCE: {s_name}]\n{d}\n"
            else:
                context += "\n(No relevant text found in local books for this specific query.)\n"
        except Exception as e:
            st.error(f"Search Error: {e}")

    if search_mode in ["Ask Mostafa Al-Adawi", "Hybrid (Both)"]:
        try:
            search = VideosSearch(f"{query} @ftawamostafaaladwy", limit=1)
            for v in search.result().get('result', []):
                try:
                    t = YouTubeTranscriptApi.get_transcript(v['id'], languages=['ar', 'en'])
                    yt_sources.append({"title": v['title'], "link": v['link']})
                    context += f"\n[VIDEO: {v['title']}]\n{' '.join([x['text'] for x in t])[:1200]}\n"
                except: continue
        except: pass
                
    return context, pdf_sources, yt_sources

# --- 5. UI ---
st.title("🕌 Haikal AI")

with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio("Search Mode:", ["Search Local Books Only", "Ask Mostafa Al-Adawi", "Hybrid (Both)"], index=2)
    st.write(f"📊 DB Size: {collection.count()} chunks")
    if st.button("🔄 Force Re-Index"):
        # This is a 'Panic Button' to clear and restart
        st.cache_resource.clear()
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        context, pdfs, vids = get_data(prompt, mode)
        st.session_state.current_sources = {"pdfs": pdfs, "vids": vids}
        
        sys_instruct = (
            "You are a scholarly assistant. Reply in the user's language. "
            "Use the context below. If context is empty, say you couldn't find "
            "specific details in the library but will answer from general knowledge."
        )
        
        try:
            # Use gemini-2.5-flash (Ensure this is in your region's quota)
            response = client_gemini.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"Context:\n{context}\n\nQuestion: {prompt}",
                config=types.GenerateContentConfig(system_instruction=sys_instruct)
            )
            answer = response.text
        except:
            answer = "I'm having trouble reaching the AI. Please try again in a moment."

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

with st.sidebar:
    st.subheader("📚 Sources:")
    for p in set(st.session_state.current_sources["pdfs"]): st.write(f"📖 {p}")
    for v in st.session_state.current_sources["vids"]: st.markdown(f"🎥 [{v['title']}]({v['link']})")
