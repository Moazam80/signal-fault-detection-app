"""
agent.py
--------
AI Agent layer using Google Gemini API.

Takes the structured classification result and asks Gemini to produce:
  1. A plain-English explanation for an engineer
  2. An urgency assessment
  3. Recommended next actions

Set your API key in the GEMINI_API_KEY environment variable.

Install:  pip install google-genai
"""

import os
import json

# We use google-genai (the new unified SDK)
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def get_ai_explanation(classification_result: dict) -> dict:
    """
    Call Gemini to explain the diagnostic result in plain English.

    Parameters
    ----------
    classification_result : output dict from classifier.classify()

    Returns
    -------
    dict with keys:
        explanation  (str) - natural language explanation
        urgency      (str) - "Low" | "Medium" | "High" | "Critical"
        actions      (list of str) - recommended next steps
        available    (bool) - False if Gemini key not set
    """
    if not GEMINI_AVAILABLE:
        return _fallback(classification_result, reason="google-genai not installed")

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return _fallback(classification_result, reason="GEMINI_API_KEY not set")

    cls      = classification_result["predicted_class"]
    feats    = classification_result["features"]
    interp   = classification_result["interpretation"]
    scores   = classification_result["scores"]
    warnings = classification_result.get("warnings", [])

    # Build scores string for context
    scores_text = ", ".join(f"{k}: {v:.2f}" for k, v in scores.items())

    prompt = f"""You are an expert machine condition monitoring engineer.
A KNN machine learning model has just analysed a vibration signal from an industrial machine.

CLASSIFICATION RESULT
---------------------
Predicted fault class : {cls}
Classifier scores     : {scores_text}

EXTRACTED FEATURES
------------------
RMS                   : {feats["rms"]}
Peak                  : {feats["peak"]}
Crest Factor          : {feats["crest_factor"]}
Fundamental Frequency : {feats["fundamental_freq_hz"]} Hz
2nd Harmonic Ratio    : {feats["second_harmonic_ratio"]}

RULE-BASED OBSERVATIONS
------------------------
{chr(10).join("- " + o for o in interp["observations"])}

WARNINGS
--------
{chr(10).join("- " + w for w in warnings) if warnings else "None"}

Please respond in JSON with exactly these three keys:
{{
  "explanation": "<2-4 sentence plain-English explanation of what this fault means for the machine and why the features indicate it>",
  "urgency": "<one of: Low | Medium | High | Critical>",
  "actions": ["<action 1>", "<action 2>", "<action 3>"]
}}
Only output valid JSON. No markdown, no extra text."""

    import time
    last_err = ""
    for attempt in range(3):
        try:
            client   = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2),
            )
            raw = response.text.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed = json.loads(raw)
            parsed["available"] = True
            return parsed

        except Exception as e:
            last_err = str(e)
            if "503" in last_err or "UNAVAILABLE" in last_err:
                time.sleep(2 ** attempt)   # 1s -> 2s -> 4s
                continue
            break   # non-retryable error

    return _fallback(classification_result, reason=last_err)



def _fallback(result: dict, reason: str = "") -> dict:
    """
    Rule-based fallback explanation when Gemini is unavailable.
    Mirrors the MATLAB interpretation layer logic.
    """
    cls = result["predicted_class"].lower().strip()
    summary = result["interpretation"]["summary"]

    urgency_map = {
        "normal":          ("Low",    ["Continue normal monitoring schedule.",
                                       "Log this result for trend tracking.",
                                       "No immediate action required."]),
        "high vibration":  ("Medium", ["Reduce machine load if possible.",
                                       "Inspect mounting and balancing.",
                                       "Schedule maintenance within 1-2 weeks."]),
        "harmonic fault":  ("Medium", ["Inspect for misalignment or looseness.",
                                       "Check coupling and shaft alignment.",
                                       "Schedule a detailed inspection."]),
        "impulsive fault": ("High",   ["Inspect bearings for defects immediately.",
                                       "Check for gear tooth damage.",
                                       "Consider taking machine offline for inspection."]),
    }

    urgency, actions = urgency_map.get(cls, ("Unknown", ["Consult a specialist."]))

    explanation = summary
    if reason:
        explanation += f" (AI explanation unavailable: {reason})"

    return {
        "explanation": explanation,
        "urgency":     urgency,
        "actions":     actions,
        "available":   False,
    }

