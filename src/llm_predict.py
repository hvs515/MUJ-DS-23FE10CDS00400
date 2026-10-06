"""
LLM-based political bias prediction using the Google Gemini API.
Reads the API key from the GEM_API environment variable (or a .env file).
"""

import os
import argparse
import json
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import errors
from pydantic import BaseModel


# Tried in order; falls back to the next if a model is overloaded or unavailable
MODELS = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"]

SYSTEM_PROMPT = (
    "You are a media bias analyst. Classify the political bias of the news article "
    "the user provides into exactly one of: FAR_LEFT, LEFT, CENTER, RIGHT, FAR_RIGHT. "
    "Judge the framing, word choice, sourcing and emphasis of the article itself, "
    "not the topic. Give a confidence between 0 and 1 and a short explanation."
)

# Load GEM_API from the project's .env file
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class BiasResult(BaseModel):
    prediction: Literal["FAR_LEFT", "LEFT", "CENTER", "RIGHT", "FAR_RIGHT"]
    confidence: float
    explanation: str


def predict_with_llm(text, client=None):
    """
    Classify an article's political bias with Gemini.

    Args:
        text: article text string
        client: optional genai.Client

    Returns:
        dict with prediction, confidence and explanation
    """
    if client is None:
        api_key = os.environ.get("GEM_API")
        if not api_key:
            raise RuntimeError("GEM_API is not set. Add it to the .env file in the project root.")
        client = genai.Client(api_key=api_key)

    config = {
        "system_instruction": SYSTEM_PROMPT,
        "response_mime_type": "application/json",
        "response_schema": BiasResult,
    }

    for model in MODELS:
        try:
            response = client.models.generate_content(model=model, contents=text, config=config)
            break
        except (errors.ServerError, errors.ClientError) as e:
            if model == MODELS[-1]:
                raise
            print(f"{model} unavailable ({e.code}), trying next model...")

    if response.parsed is None:
        return {"prediction": "UNCERTAIN", "confidence": 0.0,
                "explanation": "The model did not return a classification."}

    return response.parsed.model_dump()


def main():
    """CLI interface."""
    parser = argparse.ArgumentParser(description='Classify political bias with Gemini')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text', type=str, help='Article text to classify')
    input_group.add_argument('--file', type=str, help='Path to article text file')
    args = parser.parse_args()

    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = args.text

    print(json.dumps(predict_with_llm(text), indent=2))


if __name__ == "__main__":
    main()
