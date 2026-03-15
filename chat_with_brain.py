from youtubesearchpython import VideosSearch
from youtube_transcript_api import YouTubeTranscriptApi
from google import genai
import time

# --- CONFIG ---
API_KEY = "AIzaSyCcsgZ35VX01FjexAPf2dTJ5fPHFvS0rYA"
MODEL_NAME = "gemini-3-flash-preview"
# The specific scholar's handle
CHANNEL_HANDLE = "@ftawamostafaaladwy"

client_gemini = genai.Client(api_key=API_KEY)

def get_live_youtube_context(query):
    print(f"🔍 Searching YouTube for: '{query}' on {CHANNEL_HANDLE}...")
    
    # 1. SEARCH: We force the search to stay on the specific channel
    # Limit to top 2 videos to keep it fast
    search_query = f"{query} channel:{CHANNEL_HANDLE}"
    videos_search = VideosSearch(search_query, limit=2)
    results = videos_search.result()['result']
    
    if not results:
        return "No relevant videos found on the channel."

    context_text = ""
    for video in results:
        v_id = video['id']
        v_title = video['title']
        v_link = video['link']
        
        print(f"📥 Fetching transcript for: {v_title}...")
        try:
            # 2. TRANSCRIPT: Get Arabic/English text
            transcript = YouTubeTranscriptApi.get_transcript(v_id, languages=['ar', 'en'])
            full_text = " ".join([t['text'] for t in transcript])
            
            # Add a snippet of the transcript to the context (Gemini doesn't need 1 hour of text)
            context_text += f"\nSOURCE VIDEO: {v_title}\nLINK: {v_link}\nCONTENT: {full_text[:6000]}\n"
        except Exception:
            print(f"⚠️ Could not get transcript for: {v_title}")
            continue
            
    return context_text

def ask_ai(user_query):
    # A. Get context from YouTube
    yt_context = get_live_youtube_context(user_query)
    
    # B. Construct the final prompt
    prompt = f"""
    You are an expert assistant for the scholar Mustafa Al-Adawi.
    Use the provided YouTube transcripts to answer the question.

    RULES:
    1. Answer in the language the user used (Arabic or English).
    2. Cite the video title and provide the link at the end.
    3. If the transcript doesn't contain the answer, say you couldn't find it in the recent videos.

    CONTEXT FROM VIDEOS:
    {yt_context}

    USER QUESTION:
    {user_query}
    """

    # C. Send to Gemini
    response = client_gemini.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    return response.text

# --- RUN LOOP ---
print(f"🚀 Live Search App Active (Channel: {CHANNEL_HANDLE})")
while True:
    q = input("\nAsk a question: ")
    if q.lower() in ['quit', 'exit']: break
    
    print("⏳ Processing (this takes 5-10 seconds)...")
    try:
        answer = ask_ai(q)
        print("\n" + "="*60)
        print(answer)
        print("="*60)
    except Exception as e:
        print(f"❌ Error: {e}")