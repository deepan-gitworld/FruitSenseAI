"""Gemini Explainer module for FruitSenseAI.

Converts structured CNN and VLM evidence into concise, shopper-friendly quality explanations
using the official google-genai SDK.
"""

from __future__ import annotations

import json
import re
from typing import Any

from google import genai

from sarvamai import SarvamAI

from src.config import (
    DEFAULT_GEMINI_MODEL,
    fruit_family_name,
    get_gemini_api_key,
    get_sarvam_api_key,
)

def build_normalized_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Extract and normalize evidence fields for the prompt."""
    return {
        "predicted_fruit_condition": evidence.get("cnn_predicted_class") or evidence.get("predicted_fruit_condition"),
        "model_confidence": evidence.get("cnn_confidence") or evidence.get("model_confidence"),
        "class_probabilities": evidence.get("cnn_class_probabilities") or evidence.get("class_probabilities"),
        "visual_description": evidence.get("vlm_description") or evidence.get("visual_description"),
        "models_disagree": evidence.get("fruit_label_disagreement_heuristic") or evidence.get("models_disagree"),
        "review_status": evidence.get("review_status_heuristic") or evidence.get("review_status"),
    }

def generate_fallback_explanation(evidence: dict[str, Any]) -> str:
    """Generate a clean rule-based explanation when Gemini API is unavailable."""
    norm = build_normalized_evidence(evidence)
    pred_class = (norm["predicted_fruit_condition"] or "").lower()
    description = (norm["visual_description"] or "").lower()
    disagree = norm["models_disagree"]
    review_status = norm["review_status"]

    fruit = fruit_family_name(pred_class) or "fruit"
    if fruit != "unknown":
        fruit = fruit.removesuffix("s") if fruit.endswith("s") else fruit

    if disagree:
        return f"The condition of this {fruit} is unclear because the visible appearance conflicts with the classification. Inspect it carefully before consuming."

    if "rotten" in pred_class:
        if "mold" in description:
            return f"There may be mold on the surface of this {fruit}. Check that area carefully and avoid eating if spoiled."
        elif any(w in description for w in ["bruis", "spot", "discolor", "dark"]):
            return f"This {fruit} shows visible signs of surface damage or discoloration. Check for any soft or spoiled areas before eating."
        else:
            return f"This {fruit} shows signs of spoilage or deterioration. Inspect it closely before eating."
    else:
        # Fresh
        if any(w in description for w in ["mold", "bruis", "rot"]):
            return f"The {fruit} mostly appears fresh, but visible surface blemishes were noted. Check for soft spots before eating."
        else:
            return f"The {fruit} looks fresh, with an even surface and good color. Check for any soft spots before eating."

def generate_gemini_explanation(
    evidence: dict[str, Any],
    model_name: str = DEFAULT_GEMINI_MODEL,
) -> str:
    """Generate a concise, natural fruit quality explanation via Gemini API.

    Raises EnvironmentError if GEMINI_API_KEY is missing.
    """
    api_key = get_gemini_api_key()
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not configured in the environment or .env file.\n"
            "Set it in PowerShell with: $env:GEMINI_API_KEY=\"your_key_here\""
        )

    norm_evidence = build_normalized_evidence(evidence)
    evidence_json = json.dumps(norm_evidence, ensure_ascii=False, default=str)

    prompt = f"""
You are the fruit-quality assistant in a practical app.

The user cares about whether the fruit looks fresh, damaged, or possibly
spoiled. Explain the result as a normal person would speak to another person.

EVIDENCE FROM THE EXISTING ANALYSIS:
{evidence_json}

STRICT OUTPUT RULES:
1. Output only one or two short sentences, maximum 45 words.
2. Use everyday language, not technical or scientific report language.
3. Start with the likely fruit condition.
4. Mention only the most relevant visible signs or concern.
5. If the visual description suggests mold, say something like:
   "There may be mold on the surface, so check that area carefully."
6. If bruises, soft spots, dark spots, or unusual discoloration are
   described, mention the relevant issue naturally.
7. Never invent mold, bruises, spots, or other defects that are not
   supported by the evidence.
8. If no visible problem is described, say the fruit looks fresh based
   on its visible appearance.
9. If the prediction and visual description conflict, say the condition
   is unclear and briefly explain why.
10. Do not mention CNN, VLM, Gemini, AI, confidence percentages,
    probabilities, JSON, Grad-CAM, or model agreement.
11. Do not say the fruit is guaranteed fresh or safe to eat.
12. Do not add greetings, headings, introductions, generic disclaimers,
    long advice, or unnecessary explanations.
13. Do not describe what you are doing. Return only the final sentences
    that should appear on screen and be spoken aloud.

Examples of the desired style:

"The apple looks fresh, with a smooth red surface. Check for any soft
spots before eating."

"There may be mold on the surface, so check that area carefully. If you
find mold, avoid eating the fruit."

"The fruit's condition is unclear because its appearance does not match
the prediction. Check it carefully for signs of spoilage."
""".strip()

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={
            "temperature": 0.2,
            "max_output_tokens": 300,
            "thinking_config": {"thinking_budget": 0},
        },
    )

    explanation = (response.text or "").strip()
    explanation = explanation.strip(" \t\n\"'")
    explanation = re.sub(
        r"^(Explanation|Result|Fruit quality)\s*:\s*",
        "",
        explanation,
        flags=re.IGNORECASE,
    )

    if not explanation:
        raise RuntimeError("Gemini returned an empty response.")

    return explanation

def translate_explanation(
    text: str,
    target_language_code: str = "en-IN",
) -> str:
    """Translate the explanation with Sarvam into the target language for Sarvam TTS.

    Args:
        text: Explanation text in English.
        target_language_code: 'en-IN', 'ta-IN', or 'hi-IN'.
    Returns:
        Translated text in target language's native script.
    """
    if not text or target_language_code == "en-IN":
        return text

    language_map = {
        "ta-IN": "Tamil",
        "hi-IN": "Hindi",
    }
    lang_name = language_map.get(target_language_code)
    if not lang_name:
        return text

    api_key = get_sarvam_api_key()
    if not api_key:
        raise EnvironmentError(
            "SARVAM_API_KEY is not configured. Add it to the project .env file "
            "and restart the app before generating Tamil or Hindi speech."
        )

    try:
        client = SarvamAI(api_subscription_key=api_key)
        response = client.text.translate(
            input=text[:1000],
            source_language_code="en-IN",
            target_language_code=target_language_code,
        )
        translated = (getattr(response, "translated_text", None) or "").strip()
        if not translated:
            raise RuntimeError(f"Sarvam returned an empty {lang_name} translation.")
        return translated
    except Exception as exc:
        raise RuntimeError(f"Sarvam could not translate the explanation into {lang_name}: {exc}") from exc

