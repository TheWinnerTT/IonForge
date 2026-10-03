"""Optional (first cut if late): spoken round briefings with ElevenLabs.

Mistral Small summarizes the round in ~50 words -> ElevenLabs TTS -> mp3 in
Supabase Storage (bucket `briefings`, public) -> audio_url on a `briefing` event.
"""
import json
import os

import requests
from dotenv import load_dotenv

from integrations.llm import mistral_chat
from integrations.supabase_sync import db

load_dotenv()


def summarize_round(round_record):
    return mistral_chat(
        "You are the voice of an autonomous materials lab. Summarize the round in at most 50 spoken words: "
        "what was tested, what was learned, what happens next. Plain sentences, no lists, no formulas with subscripts.",
        json.dumps(round_record, default=str), max_tokens=200,
    ).strip()


def tts(text):
    voice = os.getenv("ELEVENLABS_VOICE_ID") or "JBFqnCBsd6RMkjVDRZzb"
    r = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice}",
        headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"], "Content-Type": "application/json"},
        json={"text": text, "model_id": os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")},
        timeout=120,
    )
    r.raise_for_status()
    return r.content


def upload_audio(path, data):
    url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY")
    r = requests.post(
        f"{url}/storage/v1/object/briefings/{path}",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "audio/mpeg", "x-upsert": "true"},
        data=data, timeout=60,
    )
    r.raise_for_status()
    return f"{url}/storage/v1/object/public/briefings/{path}"


def brief_round(run_id, round_, round_record):
    text = summarize_round(round_record)
    audio = tts(text)
    audio_url = upload_audio(f"{run_id}/round-{round_}.mp3", audio) if db.enabled else None
    db.event(run_id, round_, "lab_voice", "briefing", text, {"chars": len(text)}, audio_url=audio_url)
    return text, audio_url
