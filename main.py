import os
import re
import time
import requests
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    IpBlocked,
    RequestBlocked
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
    description="Fetch YouTube transcript and optionally generate summary via Hugging Face Inference API"
)

# -----------------------------
# Request model
# -----------------------------
class Req(BaseModel):
    url: str
    max_retries: int = 3
    retry_delay: int = 2
    summarize: bool = True


# -----------------------------
# Helpers
# -----------------------------
def extract_video_id(url: str) -> str:
    url = (url or "").strip()
    # supports watch?v=ID&list=..., shorts/ID, youtu.be/ID, embed/ID
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/|/v/)([0-9A-Za-z_-]{11})", url)
    if not m:
        raise ValueError("Could not extract video ID from URL")
    return m.group(1)


def fetch_transcript(video_id: str, max_retries: int, retry_delay: int) -> str:
    for attempt in range(max_retries):
        try:
            tl = YouTubeTranscriptApi.list_transcripts(video_id)

            # Prefer English
            try:
                transcript = tl.find_transcript(["en"])
            except Exception:
                transcript = tl.find_generated_transcript(["en"])

            data = transcript.fetch()
            return TextFormatter().format_transcript(data)

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
    """
    Summarize using Hugging Face Inference API.
    Requires HF_TOKEN.
    """
    if not ENABLE_SUMMARY:
        return ""

    if not HF_TOKEN:
        return "(Summary disabled: HF_TOKEN not set in Railway variables)"

    # Keep size limited for inference
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) > 5000:
        text = text[:5000]

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": text,
        "parameters": {"max_length": 180, "min_length": 60, "do_sample": False}
    }

    r = requests.post(HF_API_URL, headers=headers, json=payload, timeout=180)

    # HF sometimes returns a dict with error
    try:
        data = r.json()
    except Exception:
        raise RuntimeError(f"HF returned non-JSON. Status={r.status_code}. Body={r.text[:300]}")

    # Error format
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"Hugging Face error: {data['error']}")

    # Success format: list of dicts
    if isinstance(data, list) and len(data) > 0:
        return data[0].get("summary_text", "")

    return ""


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {
        "message": "YouTube Transcript API",
        "endpoints": {
            "POST /summarize": "JSON body: {url, max_retries, retry_delay, summarize}",
            "GET /health": "Health check"
        },
        "summary_enabled": ENABLE_SUMMARY,
        "hf_model": HF_SUMMARY_MODEL,
        "hf_token_set": bool(HF_TOKEN)
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/summarize")
def summarize(req: Req):
    try:
        video_id = extract_video_id(req.url)
        transcript = fetch_transcript(video_id, req.max_retries, req.retry_delay)

        summary = ""
        if req.summarize:
            summary = hf_summarize(transcript)

        return {
            "video_id": video_id,
            "transcript": transcript,
            "summary": summary
        }

    except Exception as e:
        return {"error": str(e)}
