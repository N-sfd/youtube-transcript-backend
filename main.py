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

app = FastAPI(
    title="YouTube Transcript API",
    description="Fetch transcript + optional summary via Hugging Face Inference API",
    version="1.0.0",
)

# -----------------------------
# ENV VARS (Railway Variables)
# -----------------------------
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_SUMMARY_MODEL = os.getenv("HF_SUMMARY_MODEL", "facebook/bart-large-cnn").strip()
ENABLE_SUMMARY = os.getenv("ENABLE_SUMMARY", "1") == "1"


class Req(BaseModel):
    url: str
    max_retries: int = 3
    retry_delay: int = 2
    summarize: bool = True


def extract_video_id(url: str) -> str:
    url = (url or "").strip()
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/|/v/)([0-9A-Za-z_-]{11})", url)
    if not m:
        raise ValueError("Invalid YouTube URL (could not extract video id).")
    return m.group(1)


def fetch_transcript(video_id: str, max_retries: int, retry_delay: int) -> str:
    for attempt in range(max_retries):
        try:
            tl = YouTubeTranscriptApi.list_transcripts(video_id)

            try:
                t = tl.find_transcript(["en"])
            except Exception:
                t = tl.find_generated_transcript(["en"])

            fetched = t.fetch()  # list[dict]
            return TextFormatter().format_transcript(fetched)

        except (IpBlocked, RequestBlocked):
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise ConnectionError("Blocked by YouTube (rate limit / IP blocked).")

        except TranscriptsDisabled:
            raise ValueError("Transcripts are disabled for this video.")
        except NoTranscriptFound:
            raise ValueError("No transcript found for this video.")
        except VideoUnavailable:
            raise ValueError("Video unavailable.")
        except Exception as e:
            raise RuntimeError(str(e))


def hf_summarize(text: str) -> str:
    """Uses Hugging Face Inference API."""
    if not HF_TOKEN:
        return "(HF_TOKEN not set in Railway Variables)"

    text = (text or "").strip()
    if not text:
        return "(Empty transcript)"

    url = f"https://api-inference.huggingface.co/models/{HF_SUMMARY_MODEL}"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": text[:6000],
        "parameters": {"max_length": 180, "min_length": 60, "do_sample": False},
    }

    r = requests.post(url, headers=headers, json=payload, timeout=180)

    try:
        data = r.json()
    except Exception:
        return f"(HF returned non-JSON. Status={r.status_code}. Body={r.text[:200]})"

    if isinstance(data, dict) and "estimated_time" in data:
        return "(HF model is loading. Try again in ~20 seconds.)"

    if isinstance(data, dict) and data.get("error"):
        return f"(HF error: {data['error']})"

    if isinstance(data, list) and len(data) > 0:
        return data[0].get("summary_text", "(HF: summary_text missing)")

    return f"(Unexpected HF response: {str(data)[:200]})"


@app.get("/")
def root():
    return {
        "message": "YouTube Transcript API",
        "endpoints": {
            "GET /summarize?url=...": "Returns transcript + summary",
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
        video_id = extract_video_id(req.url)
        transcript = fetch_transcript(video_id, req.max_retries, req.retry_delay)

        summary = ""
        if ENABLE_SUMMARY and req.summarize:
            summary = hf_summarize(transcript)

        return {"video_id": video_id, "transcript": transcript, "summary": summary}

    except Exception as e:
        return {"error": str(e)}


@app.get("/summarize")
def summarize_get(url: str, max_retries: int = 3, retry_delay: int = 2, summarize: bool = True):
    try:
        video_id = extract_video_id(url)
        transcript = fetch_transcript(video_id, max_retries, retry_delay)

        summary = ""
        if ENABLE_SUMMARY and summarize:
            summary = hf_summarize(transcript)

        return {"video_id": video_id, "transcript": transcript, "summary": summary}

    except Exception as e:
        return {"error": str(e)}
