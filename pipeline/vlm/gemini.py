"""Owner V. Gemini provider: one structured-JSON call per area, key in .env as GEMINI_API_KEY.

A provider is a function call(images, system, text, schema) -> (answer dict, usage dict). The Claude
provider will have the same shape.
"""
import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from pipeline.common.config import GEMINI_MODEL, ROOT

PROVIDER = "gemini"
MODEL = GEMINI_MODEL


def make_call():
    """Return the Gemini call function, or None when there is no API key."""
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    client = genai.Client(api_key=api_key)

    def call(images: list[bytes], system: str, text: str, schema: dict) -> tuple[dict, dict]:
        """images are PNG bytes. Raises on API errors and on answers that are not JSON."""
        response = client.models.generate_content(
            model=MODEL,
            contents=[*(types.Part.from_bytes(data=image, mime_type="image/png") for image in images), text],
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
