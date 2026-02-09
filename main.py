import re
import time
from fastapi import FastAPI
from pydantic import BaseModel

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from youtube_transcript_api._errors import (
    TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, IpBlocked, RequestBlocked
)

app = FastAPI(title="YouTube Transcript API", version="1.0.0")

class Req(BaseModel):
    url: str
    max_retries: int = 3
    retry_delay: int = 2

def extract_video_id(url: str) -> str:
    url = (url or "").strip()
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/|/v/)([0-9A-Za-z_-]{11})", url)
    if not m:
        raise ValueError("Invalid YouTube URL")
    return m.group(1)

def fetch_transcript(video_id: str, max_retries: int, retry_delay: int) -> str:
    for attempt in range(max_retries):
        try:
            api = YouTubeTranscriptApi()
            fetched = api.fetch(video_id)
            return TextFormatter().format_transcript(fetched)
        except (IpBlocked, RequestBlocked):
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise ConnectionError("Blocked by YouTube.")
        except TranscriptsDisabled:
            raise ValueError("Transcripts are disabled for this video.")
        except NoTranscriptFound:
            raise ValueError("No transcript found for this video.")
        except VideoUnavailable:
            raise ValueError("Video unavailable.")
        except Exception as e:
            raise RuntimeError(str(e))

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/summarize")
def summarize_post(req: Req):
    try:
        video_id = extract_video_id(req.url)
        transcript = fetch_transcript(video_id, req.max_retries, req.retry_delay)
        return {"video_id": video_id, "transcript": transcript}
    except Exception as e:
        return {"error": str(e)}

@app.get("/summarize")
def summarize_get(url: str, max_retries: int = 3, retry_delay: int = 2):
    try:
        video_id = extract_video_id(url)
        transcript = fetch_transcript(video_id, max_retries, retry_delay)
        return {"video_id": video_id, "transcript": transcript}
    except Exception as e:
        return {"error": str(e)}
