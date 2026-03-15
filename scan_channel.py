import chromadb
from chromadb.utils import embedding_functions
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
import os

# --- SETTINGS ---
DB_PATH = "./my_db"
CHANNEL_URL = "https://www.youtube.com/@ftawamostafaaladwy"
LABEL = "مصطفي العدوي" # Everything from this channel gets this label
VIDEO_LIMIT = 20  # You can increase this to 100 or 500 once you're ready

# 1. CONNECT TO YOUR BRAIN
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(
    name="religious_knowledge", 
    embedding_function=embedding_func
)

def get_channel_videos(url):
    print(f"🔍 Scanning channel for the last {VIDEO_LIMIT} videos...")
    ydl_opts = {
        'extract_flat': True, 
        'playlist_items': f'1-{VIDEO_LIMIT}'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info['entries']

def process_channel():
    # Get the list of existing IDs so we don't process duplicates
    existing_ids = collection.get()['ids']
    
    videos = get_channel_videos(CHANNEL_URL)
    
    for video in videos:
        video_id = video['id']
        video_title = video['title']
        unique_id = f"yt_{video_id}"

        # Skip if already in the database
        if unique_id in existing_ids:
            print(f"⏩ Skipping (already indexed): {video_title}")
            continue
        
        print(f"\n🎬 Processing: {video_title}")
        
        try:
            # Get Transcript
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['ar', 'en'])
            full_text = " ".join([entry['text'] for entry in transcript_list])

            # Save to Database with the Label
            source_tag = f"YouTube > {LABEL} > {video_title}"
            
            collection.add(
                ids=[unique_id],
                documents=[full_text],
                metadatas=[{"source": source_tag, "page": "Video"}]
            )
            print(f"✅ Added to brain.")

        except Exception:
            print(f"⚠️ Skipped: No transcript/captions found.")

if __name__ == "__main__":
    process_channel()