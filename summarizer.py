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
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro",
]

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _api_url(model: str, api_key: str) -> str:
    return f"{GEMINI_BASE}/{model}:generateContent?key={api_key}"


def _list_models(api_key: str) -> list[str]:
    """Return available model names for this API key."""
    try:
        r = requests.get(
            f"https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": api_key},
            timeout=15,
        )
        if r.status_code == 200:
            data = r.json()
            return [m["name"].split("/")[-1] for m in data.get("models", [])
                    if "generateContent" in m.get("supportedGenerationMethods", [])]
    except Exception:
        pass
    return []


def resolve_model(api_key: str, requested_model: str | None = None) -> str | None:
    """
    Return a usable model name, or None if none can be found.

    If `requested_model` is supplied and valid, use it.
    Otherwise fall back through PREFERRED_MODELS.
    """
    available = _list_models(api_key)
    if not available:
        # Can't verify – return the requested model or first preferred
        return requested_model or PREFERRED_MODELS[0]

    if requested_model and requested_model in available:
        return requested_model

    for m in PREFERRED_MODELS:
        if m in available:
            return m

    # Last resort: return anything available
    return available[0] if available else None


# ---------------------------------------------------------------------------
# Core summariser
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

    Parameters
    ----------
    person  : GitHub login
    counts  : dict of activity type → count  e.g. {"commit": 12, "pull_request": 3}
    titles  : list of recent activity titles (commits, PRs, issues)
    api_key : Gemini API key
    model   : Gemini model name

    Returns
    -------
    A short summary string, or an error message starting with "Error:".
    """
    if not api_key:
        return "Error: Gemini API key not configured."
    if not model:
        return "Error: No suitable Gemini model found."

    # Build human-readable counts
    count_lines = "\n".join(
        f"  - {k.replace('_', ' ').title()}: {v}" for k, v in counts.items() if v > 0
    )
    # Sanitise titles: take up to 15, strip empty strings
    clean_titles = [t.strip() for t in titles if t and t.strip()][:15]
    title_lines = "\n".join(f"  - {t}" for t in clean_titles) if clean_titles else "  (no titles available)"

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
        "generationConfig": {
            "maxOutputTokens": 200,
            "temperature": 0.2,
        },
    }

    try:
        r = requests.post(
            _api_url(model, api_key),
            json=payload,
            timeout=30,
        )
    except requests.exceptions.Timeout:
        return "Error: Gemini API request timed out."
    except requests.exceptions.ConnectionError:
        return "Error: Could not reach Gemini API. Check your internet connection."

    if r.status_code == 400:
        return f"Error: Bad request to Gemini API – {r.json().get('error', {}).get('message', 'unknown error')}."
    if r.status_code == 401 or r.status_code == 403:
        return "Error: Gemini API key is invalid or access is denied."
    if r.status_code == 429:
        return "Error: Gemini API quota exceeded. Try again later."
    if r.status_code != 200:
        return f"Error: Gemini API returned HTTP {r.status_code}."

    try:
        data = r.json()
        candidates = data.get("candidates", [])
        if not candidates:
            # Check for prompt feedback (blocked content)
            feedback = data.get("promptFeedback", {})
            reason = feedback.get("blockReason", "unknown")
            return f"Error: Gemini returned no response (block reason: {reason})."
        text = candidates[0]["content"]["parts"][0]["text"].strip()
        return text if text else "Error: Gemini returned an empty response."
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        return f"Error: Could not parse Gemini response ({exc})."
