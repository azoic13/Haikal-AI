# --- STEP 0: THE SQLITE FIX (MUST BE AT THE VERY TOP) ---
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import streamlit_analytics2 as streamlit_analytics
import chromadb
from chromadb.utils import embedding_functions
from youtubesearchpython import VideosSearch
from youtube_transcript_api import YouTubeTranscriptApi
from google import genai
from google.genai import types
from fpdf import FPDF
import arabic_reshaper
from bidi.algorithm import get_display
import os
import zipfile
import pathlib

# ---------------------------------------------------------------------------
# STEP 1 — RESOLVE PATHS RELATIVE TO THIS FILE
# Works correctly locally AND on Streamlit Cloud no matter the CWD.
# ---------------------------------------------------------------------------
BASE_DIR = pathlib.Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# STEP 2 — EXTRACT my_db.zip ON FIRST RUN
# Streamlit Cloud's repo filesystem is read-only, so we extract into /tmp
# which is always writable. On subsequent runs the folder already exists
# so this block is skipped (fast).
# ---------------------------------------------------------------------------
DB_ZIP    = BASE_DIR / "my_db.zip"
DB_FOLDER = pathlib.Path("/tmp/my_db")   # writable on every platform

if not DB_FOLDER.exists():
    if DB_ZIP.exists():
        with zipfile.ZipFile(DB_ZIP, "r") as zf:
            zf.extractall("/tmp")         # produces /tmp/my_db/
    else:
        # No zip present — ChromaDB will just create an empty collection.
        DB_FOLDER.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# CONSTANTS — single source of truth for mode label strings.
# The radio widget and get_data() both reference these, so they can never
# drift out of sync (that was the original search bug).
# ---------------------------------------------------------------------------
MODE_BOOKS   = "Search Hadith Books (كتب الاحاديث و التفسير) Only"
MODE_YOUTUBE = "Mostafa Al-Adawi Youtube Channel"
MODE_HYBRID  = "Hybrid (Both)"

# ---------------------------------------------------------------------------
# MAIN APP — wrapped in analytics tracker
# ---------------------------------------------------------------------------
with streamlit_analytics.track(
    save_to_json=str(BASE_DIR / "analytics.json"),
    unsafe_password="haikal2026"
):
    st.set_page_config(page_title="Sharee'a (شريعة) AI", page_icon="🕌", layout="wide")

    # ── API key ──────────────────────────────────────────────────────────────
    if "GEMINI_API_KEY" not in st.secrets:
        st.error("Please add GEMINI_API_KEY to your Streamlit Secrets.")
        st.stop()

    client_gemini = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

    # ── Session state ────────────────────────────────────────────────────────
    if "messages"     not in st.session_state: st.session_state.messages      = []
    if "current_pdfs" not in st.session_state: st.session_state.current_pdfs  = []
    if "current_vids" not in st.session_state: st.session_state.current_vids  = []

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    # Use /tmp/my_db — writable on Streamlit Cloud
    client_db  = chromadb.PersistentClient(path=str(DB_FOLDER))
    collection = client_db.get_or_create_collection(
        name="religious_knowledge",
        embedding_function=embedding_func
    )

    # ── Font path (arial.ttf ships in the repo) ──────────────────────────────
    FONT_PATH = BASE_DIR / "arial.ttf"

    # =========================================================================
    # HELPER FUNCTIONS
    # =========================================================================

    def fix_arabic_for_pdf(text: str) -> str:
        """Reshape + apply BiDi so Arabic renders correctly in fpdf."""
        if not text:
            return ""
        return get_display(arabic_reshaper.reshape(text))

    def create_pdf(question: str, answer: str) -> bytes:
        pdf = FPDF()
        pdf.add_page()
        if FONT_PATH.exists():
            pdf.add_font("ArialAR", "", str(FONT_PATH))
            pdf.set_font("ArialAR", size=12)
        else:
            pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, txt=fix_arabic_for_pdf(f"السؤال: {question}"), align="R")
        pdf.ln(5)
        pdf.multi_cell(0, 10, txt=fix_arabic_for_pdf(f"الإجابة:\n{answer}"), align="R")
        return pdf.output()

    def get_data(query: str, search_mode: str):
        """
        Returns (combined_context, pdf_sources, yt_sources).
        Fetches from ChromaDB books and/or YouTube depending on search_mode.
        """
        pdf_context, pdf_sources = "", []
        yt_context,  yt_sources  = "", []
        CHANNELS = ["@ftawamostafaaladwy", "@fatawa_eladawy"]

        # ── Book / ChromaDB search ───────────────────────────────────────────
        if search_mode in [MODE_BOOKS, MODE_HYBRID]:
            try:
                results = collection.query(query_texts=[query], n_results=3)
                if results and results.get("documents") and results["documents"][0]:
                    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                        source = meta.get("source", "Unknown Book")
                        pdf_sources.append(source)
                        pdf_context += f"\n[BOOK SOURCE: {source}]\n{doc}\n"
                else:
                    st.warning("⚠️ No results in book database. Make sure books have been ingested.")
            except Exception as e:
                st.warning(f"⚠️ Book search error: {e}")

        # ── YouTube search ───────────────────────────────────────────────────
        # NOTE: search_mode string now matches the radio label exactly (was the bug)
        if search_mode in [MODE_YOUTUBE, MODE_HYBRID]:
            found_any = False
            for handle in CHANNELS:
                try:
                    search_result = VideosSearch(f"{query} {handle}", limit=2)
                    videos = search_result.result().get("result", [])
                    for v in videos:
                        try:
                            transcript = YouTubeTranscriptApi.get_transcript(
                                v["id"], languages=["ar", "en"]
                            )
                            title = str(v["title"])
                            link  = str(v["link"])
                            yt_sources.append({"title": title, "link": link})
                            transcript_text = " ".join(x["text"] for x in transcript)[:2000]
                            yt_context += (
                                f"\n[VIDEO SOURCE: {title}]\n"
                                f"Link: {link}\n"
                                f"Transcript: {transcript_text}\n"
                            )
                            found_any = True
                        except Exception as e:
                            st.warning(
                                f"⚠️ Transcript unavailable for '{v.get('title', 'unknown')}': {e}"
                            )
                except Exception as e:
                    st.warning(f"⚠️ YouTube search error ({handle}): {e}")

            if not found_any:
                st.warning("⚠️ No YouTube transcripts could be retrieved for this query.")

        return pdf_context + yt_context, pdf_sources, yt_sources

    # =========================================================================
    # SIDEBAR
    # =========================================================================
    with st.sidebar:
        st.title("⚙️ Control Room")

        mode = st.radio(
            "Search Mode:",
            [MODE_BOOKS, MODE_YOUTUBE, MODE_HYBRID],
            index=2
        )

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages     = []
            st.session_state.current_pdfs = []
            st.session_state.current_vids = []
            st.rerun()

        st.divider()
        st.subheader("📍 Sources Consulted:")
        for p in set(st.session_state.current_pdfs):
            st.write(f"📖 {p}")
        for v in st.session_state.current_vids:
            st.markdown(f"🎥 [{v['title']}]({v['link']})")

        st.divider()
        st.caption("Admin: Add ?analytics=on to URL. Pass: haikal2026")

    # =========================================================================
    # MAIN CHAT INTERFACE
    # =========================================================================
    st.title("🕌 Sharee'a AI (شريعة)")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching sources..."):
                context, pdfs, vids = get_data(prompt, mode)
                st.session_state.current_pdfs = pdfs
                st.session_state.current_vids = vids

                # ── Gemini call ──────────────────────────────────────────────
                try:
                    response = client_gemini.models.generate_content(
                        model="gemini-2.0-flash",   # FIX: was a non-existent model name
                        contents=f"CONTEXT:\n{context}\n\nQ: {prompt}",
                        config=types.GenerateContentConfig(
                            system_instruction=(
                                "You are an expert Islamic knowledge assistant specialising in "
                                "the teachings of Sheikh Mostafa Al-Adawi. "
                                "Always: 1) Provide a confidence score (e.g. Confidence: 85%). "
                                "2) Cite the source titles you used. "
                                "3) Reply in the same language the user used."
                            ),
                            temperature=0.3,
                        ),
                    )
                    answer_text = response.text
                except Exception as e:
                    answer_text = f"❌ Gemini API error: {e}"

                st.markdown(answer_text)
                st.session_state.messages.append({"role": "assistant", "content": answer_text})

                # ── PDF download ─────────────────────────────────────────────
                try:
                    pdf_bytes = create_pdf(prompt, answer_text)
                    st.download_button(
                        label="📥 Save as PDF",
                        data=pdf_bytes,
                        file_name="Sharee'a_Report.pdf",
                        mime="application/pdf",
                    )
                except Exception as e:
                    print(f"PDF generation error: {e}")

                # NOTE: st.rerun() intentionally removed — Streamlit reruns
                # automatically after st.chat_input; the explicit call was
                # causing UI flicker and could cut off the assistant message.
