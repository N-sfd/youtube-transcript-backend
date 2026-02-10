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

# -------------------------
# ENV (set in Railway Variables)
# -------------------------
ENABLE_SUMMARY = os.getenv("ENABLE_SUMMARY", "1") == "1"
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_SUMMARY_MODEL = os.getenv("HF_SUMMARY_MODEL", "facebook/bart-large-cnn").strip()

HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_SUMMARY_MODEL}"

app = FastAPI(
    title="YouTube Transcript API",
    version="1.0.0",
    description="Fetch YouTube transcript and optionally generate a summary via Hugging Face Inference API",
)

class Req(BaseModel):
    url: str
    max_retries: int = 3
    retry_delay: int = 2
    summarize: bool = True


def extract_video_id(url: str) -> str:
    url = (url or "").strip()
    if not url:
        raise ValueError("URL is empty")

    # Works for:
    # watch?v=ID&list=..., youtu.be/ID, shorts/ID, embed/ID
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/|/v/)([0-9A-Za-z_-]{11})", url)
    if not m:
        raise ValueError("Could not extract video ID from URL")
    return m.group(1)


def fetch_transcript_text(video_id: str, max_retries: int, retry_delay: int) -> str:
    api = YouTubeTranscriptApi()

    for attempt in range(max_retries):
        try:
            fetched = api.fetch(video_id, languages=["en"])
            transcript_data = list(fetched)  # IMPORTANT for TextFormatter
            return TextFormatter().format_transcript(transcript_data)

        except (IpBlocked, RequestBlocked):
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise ConnectionError("Blocked by YouTube (IP/Request blocked). Try later.")

        except TranscriptsDisabled:
            raise ValueError("Transcripts are disabled for this video.")
        except NoTranscriptFound:
            raise ValueError("No transcript found for this video.")
        except VideoUnavailable:
            raise ValueError("Video unavailable.")
        except Exception as e:
            raise RuntimeError(str(e))


def hf_summarize(text: str) -> str:
    if not ENABLE_SUMMARY:
        return ""

    if not HF_TOKEN:
        return ""  # summary disabled if no token

    text = (text or "").strip()
    if not text:
        return ""

    # Keep payload reasonable
    text = text[:6000]

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": text,
        "parameters": {"max_length": 170, "min_length": 60, "do_sample": False},
    }

    r = requests.post(HF_API_URL, headers=headers, json=payload, timeout=180)

    # HF sometimes returns dict with error
    try:
        data = r.json()
    except Exception:
        return ""

    if isinstance(data, dict) and data.get("error"):
        return ""

    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        return data[0].get("summary_text", "") or ""

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
def summarize(req: Req):
    try:
        video_id = extract_video_id(req.url)
        transcript = fetch_transcript_text(video_id, req.max_retries, req.retry_delay)

        summary = ""
        if req.summarize:
            summary = hf_summarize(transcript)

        return {
            "video_id": video_id,
            "transcript": transcript,
            "summary": summary,
        }

    except Exception as e:
        return {"error": str(e)}
