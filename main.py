import os
import re
import time
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

# Optional summarization
ENABLE_SUMMARY = os.getenv("ENABLE_SUMMARY", "true").lower() == "true"
HF_MODEL = os.getenv("HF_SUMMARY_MODEL", "facebook/bart-large-cnn")
HF_TOKEN = os.getenv("HF_TOKEN")

summarizer = None
if ENABLE_SUMMARY and HF_TOKEN:
    from transformers import pipeline
    summarizer = pipeline(
        "summarization",
        model=HF_MODEL,
        token=HF_TOKEN,
        device=-1
    )

app = FastAPI(
    title="YouTube Transcript API",
    version="1.0.0"
)

class RequestBody(BaseModel):
    url: str
    max_retries: int = 3
    retry_delay: int = 2
    summarize: bool = False


def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\\.be/|shorts/|embed/)([\\w-]{11})", url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    return match.group(1)


def fetch_transcript(video_id: str, retries: int, delay: int) -> str:
    for attempt in range(retries):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            return TextFormatter().format_transcript(transcript)
        except (IpBlocked, RequestBlocked):
            time.sleep(delay)
        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
            raise RuntimeError(str(e))
    raise RuntimeError("YouTube blocked this request")


@app.get("/")
def root():
    return {
        "message": "YouTube Transcript API",
        "summary_enabled": ENABLE_SUMMARY
    }


@app.post("/summarize")
def summarize(data: RequestBody):
    video_id = extract_video_id(data.url)
    transcript = fetch_transcript(video_id, data.max_retries, data.retry_delay)

    summary = None
    if data.summarize and summarizer:
        summary = summarizer(
            transcript[:4000],
            max_length=160,
            min_length=60,
            do_sample=False
        )[0]["summary_text"]

    return {
        "video_id": video_id,
        "transcript": transcript,
        "summary": summary
    }
