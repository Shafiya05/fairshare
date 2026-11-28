"""
summarizer.py – Gemini REST API integration for FairShare.

Uses the Gemini generateContent REST endpoint via the `requests` library.
No google-generativeai SDK required.
"""

import requests
import json


# ---------------------------------------------------------------------------
# Default / fallback model choices (checked in order)
# ---------------------------------------------------------------------------
PREFERRED_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-4-hydrogen",
]

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# Sentinel prefix: summarize() returns this when the model URL is simply wrong
# (HTTP 404) so app.py can reset the cached model and retry with a better one.
_MODEL_404_PREFIX = "Error:MODEL_404:"


def is_model_404(text: str) -> bool:
    """True when summarize() signalled that the model endpoint returned 404."""
    return text.startswith(_MODEL_404_PREFIX)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _api_url(model: str) -> str:
    """Return the generateContent URL (key sent via header, not URL)."""
    return f"{GEMINI_BASE}/{model}:generateContent"


def _list_models(api_key: str) -> list[str]:
    """Return available model names for this API key (empty list on any error)."""
    try:
        r = requests.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            headers={"x-goog-api-key": api_key},
            timeout=15,
        )
        if r.status_code == 200:
            data = r.json()
            return [
                m["name"].split("/")[-1]
                for m in data.get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])
            ]
    except Exception:
        pass
    return []


def resolve_model(api_key: str, requested_model: str | None = None) -> str | None:
    """
    Return a usable model name, or None if none can be found.

    * Normalises the model name to lowercase (Gemini API is case-sensitive).
    * If `requested_model` is supplied and in the available list, uses it.
    * Otherwise falls back through PREFERRED_MODELS.
    * If the available-model list cannot be fetched, returns the requested
      model (normalised) or the first PREFERRED_MODELS entry.
    """
    if requested_model:
        requested_model = requested_model.strip().lower()

    available = _list_models(api_key)

    if not available:
        # Can't verify – trust the requested model or first preferred
        return requested_model or PREFERRED_MODELS[0]

    if requested_model and requested_model in available:
        return requested_model

    for m in PREFERRED_MODELS:
        if m in available:
            return m

    return available[0] if available else None


# ---------------------------------------------------------------------------
# Internal: single-model attempt
# ---------------------------------------------------------------------------

def _call_model(model: str, api_key: str, payload: dict) -> str:
    """
    POST one generateContent request.  Returns:
      - The generated text on success.
      - An "Error:..." string on any failure.
      - The _MODEL_404_PREFIX sentinel when HTTP 404 is returned (bad model name).
    """
    try:
        r = requests.post(
            _api_url(model),
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=20,
        )
    except requests.exceptions.Timeout:
        return "Error: Gemini API request timed out."
    except requests.exceptions.ConnectionError:
        return "Error: Could not reach Gemini API. Check your internet connection."

    if r.status_code == 404:
        # Model URL is wrong – caller should try next model
        return f"{_MODEL_404_PREFIX}model '{model}' not found (HTTP 404)."
    if r.status_code == 400:
        try:
            msg = r.json().get("error", {}).get("message", "unknown error")
        except Exception:
            msg = "unknown error"
        return f"Error: Bad request to Gemini API – {msg}."
    if r.status_code in (401, 403):
        return "Error: Gemini API key is invalid or access is denied."
    if r.status_code == 429:
        return "Error: Gemini API quota exceeded. Try again later."
    if r.status_code != 200:
        return f"Error: Gemini API returned HTTP {r.status_code}."

    try:
        data = r.json()
        candidates = data.get("candidates", [])
        if not candidates:
            feedback = data.get("promptFeedback", {})
            reason = feedback.get("blockReason", "unknown")
            return f"Error: Gemini returned no response (block reason: {reason})."
        text = candidates[0]["content"]["parts"][0]["text"].strip()
        return text if text else "Error: Gemini returned an empty response."
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        return f"Error: Could not parse Gemini response ({exc})."


# ---------------------------------------------------------------------------
# Public summariser
# ---------------------------------------------------------------------------

def summarize(
    person: str,
    counts: dict,
    titles: list[str],
    api_key: str,
    model: str,
) -> str:
    """
    Generate a two-sentence contributor summary using the Gemini API.

    If the primary `model` returns HTTP 404 (bad model name / stale cache),
    automatically retries each entry in PREFERRED_MODELS before giving up.

    Returns
    -------
    A short summary string, or an error message starting with "Error:".
    The special prefix "Error:MODEL_404:" means every tried model URL was
    wrong – app.py should reset its cached model and re-resolve.
    """
    if not api_key:
        return "Error: Gemini API key not configured."
    if not model:
        return "Error: No suitable Gemini model found."

    # Normalise the model name (guards against stale mixed-case session values)
    model = model.strip().lower()

    count_lines = "\n".join(
        f"  - {k.replace('_', ' ').title()}: {v}" for k, v in counts.items() if v > 0
    )
    clean_titles = [t.strip() for t in titles if t and t.strip()][:15]
    title_lines = (
        "\n".join(f"  - {t}" for t in clean_titles)
        if clean_titles else "  (no titles available)"
    )

    prompt = f"""You are analysing GitHub contribution data for a software project.

Contributor: {person}

Activity counts:
{count_lines if count_lines else "  (no recorded activity)"}

Recent activity titles:
{title_lines}

Write exactly two concise sentences describing this contributor's activity.
Use only the data provided above. Do not infer job roles, skills, personal qualities, productivity, intentions, or achievements that are not directly supported by the numbers and titles above.
Do not guess or speculate. Be factual and brief."""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 200, "temperature": 0.2},
    }

    # Try the requested model first, then fall back through PREFERRED_MODELS
    # Build the trial order: primary first, then any preferred not already tried
    tried: list[str] = []
    trial_order: list[str] = [model] + [
        m for m in PREFERRED_MODELS if m != model
    ]

    last_error = f"Error: No working Gemini model found (tried {trial_order})."
    for candidate in trial_order:
        result = _call_model(candidate, api_key, payload)
        tried.append(candidate)
        if not result.startswith("Error:"):
            return result          # success
        if is_model_404(result):
            continue               # bad model name → try next
        return result              # real error (auth, quota, etc.) – stop immediately

    # Every candidate returned 404
    return f"{_MODEL_404_PREFIX}all tried models returned 404 ({tried})."
