import os
import re
import time
import requests
from fastapi import FastAPI
from pydantic import BaseModel

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    IpBlocked,
    RequestBlocked,
)

# -----------------------------
# ENV
# -----------------------------
ENABLE_SUMMARY = os.getenv("ENABLE_SUMMARY", "1") == "1"
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_SUMMARY_MODEL = os.getenv("HF_SUMMARY_MODEL", "facebook/bart-large-cnn").strip()
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_SUMMARY_MODEL}"

app = FastAPI(
    title="YouTube Transcript API",
    version="1.0.0",
    description="Fetch YouTube transcript and optionally generate summary via Hugging Face Inference API",
)

class Req(BaseModel):
    url: str
    max_retries: int = 3
    retry_delay: int = 2
    summarize: bool = True


def extract_video_id(url: str) -> str:
    url = (url or "").strip()
    # supports watch?v=ID&list=..., youtu.be/ID, shorts/ID, embed/ID
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/|/v/)([0-9A-Za-z_-]{11})", url)
    if not m:
        raise ValueError("Invalid YouTube URL (could not find a video id).")
    return m.group(1)


def fetch_transcript_text(video_id: str, max_retries: int, retry_delay: int) -> str:
    """
    Works across youtube-transcript-api variants:
    - Sometimes returns dicts: {"text","start","duration"}
    - Sometimes returns objects: item.text, item.start, item.duration
    """
    for attempt in range(max_retries):
        try:
            api = YouTubeTranscriptApi()
            fetched = api.fetch(video_id, languages=["en"])  # iterable

            transcript_items = []
            for item in fetched:
                # ✅ If item is already a dict
                if isinstance(item, dict):
                    transcript_items.append({
                        "text": item.get("text", ""),
                        "start": float(item.get("start", 0.0) or 0.0),
                        "duration": float(item.get("duration", 0.0) or 0.0),
                    })
                else:
                    # ✅ If item is an object with attributes
                    transcript_items.append({
                        "text": getattr(item, "text", ""),
                        "start": float(getattr(item, "start", 0.0) or 0.0),
                        "duration": float(getattr(item, "duration", 0.0) or 0.0),
                    })

            return TextFormatter().format_transcript(transcript_items)

        except (IpBlocked, RequestBlocked):
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise ConnectionError("Blocked by YouTube (IP/Request blocked). Try later.")
        except TranscriptsDisabled:
            raise ValueError("Transcripts are disabled for this video.")
        except NoTranscriptFound:
            raise ValueError("No transcript found for this video (not available in requested language).")
        except VideoUnavailable:
            raise ValueError("Video unavailable.")
        except Exception as e:
            raise RuntimeError(str(e))



def summarize_with_hf(text: str) -> str:
    if not ENABLE_SUMMARY:
        return ""

    if not HF_TOKEN:
        return ""  # silently return no summary if no token

    text = (text or "").strip()
    if not text:
        return ""

    # Keep input smaller to avoid inference limits/timeouts
    text = text[:6000]

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": text, "parameters": {"max_length": 160, "min_length": 60, "do_sample": False}}

    r = requests.post(HF_API_URL, headers=headers, json=payload, timeout=180)

    # HF sometimes returns 503 while loading model
    if r.status_code == 503:
        return ""

    r.raise_for_status()
    data = r.json()

    # Typical response: [{"summary_text": "..."}]
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0].get("summary_text", "") or data[0].get("generated_text", "") or ""
    if isinstance(data, dict):
        return data.get("summary_text", "") or data.get("generated_text", "") or ""

    return ""


@app.get("/")
def root():
    return {
        "message": "YouTube Transcript API",
        "endpoints": {
            "POST /summarize": "JSON body: {url, max_retries, retry_delay, summarize}",
            "GET /health": "Health check",
        },
        "summary_enabled": ENABLE_SUMMARY,
        "hf_model": HF_SUMMARY_MODEL,
        "hf_token_set": bool(HF_TOKEN),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/summarize")
def summarize_post(req: Req):
    try:
        vid = extract_video_id(req.url)
        transcript = fetch_transcript_text(vid, req.max_retries, req.retry_delay)
        summary = summarize_with_hf(transcript) if req.summarize else ""
        return {"video_id": vid, "transcript": transcript, "summary": summary, "error": ""}
    except Exception as e:
        # always return JSON
        return {"video_id": "", "transcript": "", "summary": "", "error": str(e)}

