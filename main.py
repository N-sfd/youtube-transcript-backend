import os
import re
import time
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

from transformers import pipeline


# ------------------ App ------------------
app = FastAPI(
    title="YouTube Transcript API",
    version="1.0"
)

# ------------------ HF Summarizer ------------------
HF_MODEL = "facebook/bart-large-cnn"
HF_TOKEN = os.getenv("HF_TOKEN")

summarizer = None
if HF_TOKEN:
    summarizer = pipeline(
        "summarization",
        model=HF_MODEL,
        token=HF_TOKEN
    )

# ------------------ Request Model ------------------
class SummarizeRequest(BaseModel):
    url: str
    max_retries: int = 3
    retry_delay: int = 2
    summarize: bool = True


# ------------------ Helpers ------------------
def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([0-9A-Za-z_-]{11})", url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    return match.group(1)


def fetch_transcript(video_id: str, retries: int, delay: int) -> str:
    for attempt in range(retries):
        try:
            fetched = YouTubeTranscriptApi.fetch(video_id, languages=["en"])
            return TextFormatter().format_transcript(list(fetched))

        except (IpBlocked, RequestBlocked):
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
                continue
            raise RuntimeError("Blocked by YouTube")

        except TranscriptsDisabled:
            raise RuntimeError("Transcripts disabled")

        except NoTranscriptFound:
            raise RuntimeError("No transcript found")

        except VideoUnavailable:
            raise RuntimeError("Video unavailable")


def summarize_text(text: str) -> Optional[str]:
    if not summarizer:
        return None

    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    summaries = summarizer(chunks, max_length=130, min_length=40, do_sample=False)
    return " ".join(s["summary_text"] for s in summaries)


# ------------------ Routes ------------------
@app.get("/")
def root():
    return {
        "message": "YouTube Transcript API",
        "summary_enabled": bool(summarizer),
        "hf_model": HF_MODEL
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/summarize")
def summarize(req: SummarizeRequest):
    try:
        video_id = extract_video_id(req.url)
        transcript = fetch_transcript(video_id, req.max_retries, req.retry_delay)

        summary = None
        if req.summarize:
            summary = summarize_text(transcript)

        return {
            "video_id": video_id,
            "transcript": transcript,
            "summary": summary
        }

    except Exception as e:
        return {"error": str(e)}
