import os
import re
import time
import requests
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    IpBlocked,
    RequestBlocked,
)

# -----------------------------
# ENV (Railway Variables)
# -----------------------------
ENABLE_SUMMARY = os.getenv("ENABLE_SUMMARY", "1") == "1"
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_SUMMARY_MODEL = os.getenv("HF_SUMMARY_MODEL", "facebook/bart-large-cnn").strip()
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_SUMMARY_MODEL}"


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
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/|/v/)([0-9A-Za-z_-]{11})", url)
    if not m:
        raise ValueError("Invalid YouTube URL (could not find video id).")
    return m.group(1)


def _normalize_items(items: List[Any]) -> List[Dict[str, Any]]:
    """
    Ensure each transcript item is a dict with keys: text/start/duration.
    Works whether the library returns dicts or objects.
    """
    normalized: List[Dict[str, Any]] = []
    for it in items:
        if isinstance(it, dict):
            normalized.append(
                {
                    "text": it.get("text", "") or "",
                    "start": float(it.get("start", 0.0) or 0.0),
                    "duration": float(it.get("duration", 0.0) or 0.0),
                }
            )
        else:
            normalized.append(
                {
                    "text": getattr(it, "text", "") or "",
                    "start": float(getattr(it, "start", 0.0) or 0.0),
                    "duration": float(getattr(it, "duration", 0.0) or 0.0),
                }
            )
    return normalized


def fetch_transcript_text(video_id: str, max_retries: int, retry_delay: int) -> str:
    """
    Most stable approach:
    - Try class method get_transcript (common)
    - If not available, try fetch() API (newer)
    - Build plain text by joining 'text' fields (NO TextFormatter).
    """
    last_err: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            # 1) Prefer get_transcript if it exists in installed version
            try:
                items = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            except AttributeError:
                # 2) Fallback: fetch() API (some versions expose this)
                api = YouTubeTranscriptApi()
                fetched = api.fetch(video_id, languages=["en"])
                items = list(fetched)

            normalized = _normalize_items(items)

            # Join lines safely (this avoids ALL ".text" issues)
            lines = [x["text"].replace("\n", " ").strip() for x in normalized if x.get("text")]
            transcript_text = "\n".join(lines).strip()

            if not transcript_text:
                raise ValueError("Transcript returned empty text.")
            return transcript_text

        except (IpBlocked, RequestBlocked) as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise ConnectionError("Blocked by YouTube (IP/Request blocked). Try later.")
        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
            raise ValueError(str(e))
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise RuntimeError(str(last_err))


def summarize_with_hf(text: str) -> str:
    if not ENABLE_SUMMARY:
        return ""

    if not HF_TOKEN:
        return ""

    text = (text or "").strip()
    if not text:
        return ""

    text = text[:6000]

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
    "inputs": text,
    "parameters": {
        "max_length": 180,
        "min_length": 60,
        "do_sample": False
    }
}


    try:
        r = requests.post(HF_API_URL, headers=headers, json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()

        if isinstance(data, list) and data:
            return data[0].get("summary_text") or data[0].get("generated_text", "")
        if isinstance(data, dict):
            return data.get("summary_text") or data.get("generated_text", "")

        return ""

    except Exception as e:
        return f"HF Inference error: {str(e)}"



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
        return {"video_id": "", "transcript": "", "summary": "", "error": str(e)}

@app.get("/summarize")
def summarize_get(url: str, max_retries: int = 3, retry_delay: int = 2, summarize: bool = True):
    req = Req(url=url, max_retries=max_retries, retry_delay=retry_delay, summarize=summarize)
    return summarize_post(req)


