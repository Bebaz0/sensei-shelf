"""Owner V. Gemini provider: one structured-JSON call per frame, key in .env as GEMINI_API_KEY.

A provider is a function call(parts, system, schema) -> (answer dict, usage dict), where parts is a list of
text (str) and PNG images (bytes) in message order. The Claude provider will have the same shape.
"""
import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from pipeline.common.config import GEMINI_MODEL, ROOT

PROVIDER = "gemini"
MODEL = GEMINI_MODEL
# Image tokens depend on this tier, not on file size: 263 / 535 / 1,064 tokens per crop for LOW / MEDIUM / HIGH
# (HIGH is the default). The crops are small, so MEDIUM keeps their detail. Part of the cache key (SETTINGS).
MEDIA_RESOLUTION = "MEDIA_RESOLUTION_MEDIUM"
SETTINGS = f"{MODEL} {MEDIA_RESOLUTION}"


def make_call():
    """Return the Gemini call function, or None when there is no API key."""
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    client = genai.Client(api_key=api_key)

    def call(parts: list[str | bytes], system: str, schema: dict) -> tuple[dict, dict]:
        """parts are text and PNG bytes. Raises on API errors and on answers that are not JSON."""
        contents = [
            types.Part.from_bytes(data=part, mime_type="image/png", media_resolution=MEDIA_RESOLUTION)
            if isinstance(part, bytes) else part
            for part in parts
        ]
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0,
                response_mime_type="application/json",
                response_json_schema=schema,
            ),
        )
        usage = response.usage_metadata
        return json.loads(response.text), {
            "input_tokens": usage.prompt_token_count or 0,
            "output_tokens": (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0),
        }

    return call
