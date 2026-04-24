"""
Legal AI Engine — HuggingFace Only
Uses Mistral-7B-Instruct (free, no credit card needed).
Get your free token at: https://huggingface.co/settings/tokens
"""

import os
import re
import logging
import requests

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_API_TOKEN", "")          # paste your HF token here or set env var
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"  # free, strong, good at legal text
HF_API    = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

SYSTEM_PROMPT = """You are a senior Pakistani legal research assistant specializing in case law 
from the Sindh High Court (SHC), Lahore High Court (LHC), and Islamabad High Court (IHC).

Given a legal query and relevant case excerpts, you must:
1. Identify the most relevant precedents
2. Explain what the courts decided in those cases
3. Extract key legal principles and holdings
4. Provide a clear analysis with proper citations

Format your response with these sections:
**Legal Issue**: (one line summary)
**Relevant Cases**: (list the most applicable cases)
**Court Decisions**: (what was decided and why)
**Legal Principles**: (key rules established)
**Analysis**: (how this applies to the current situation)
**Citations**: (full list at the end)

Be precise. Always cite court name, citation number, and year.
Note: This is legal research assistance, not legal advice."""


# ── HuggingFace Call ──────────────────────────────────────────────────────────

def ask_llm(prompt: str) -> str:
    """
    Send prompt to HuggingFace Mistral-7B and return the response text.
    Falls back to rule-based summary if token is missing or API is down.
    """
    if not HF_TOKEN:
        log.warning("HF_API_TOKEN not set — showing raw results only.")
        return _no_token_message()

    # Format as Mistral instruction template
    full_prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\n{prompt} [/INST]"

    try:
        resp = requests.post(
            HF_API,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={
                "inputs": full_prompt,
                "parameters": {
                    "max_new_tokens": 900,
                    "temperature": 0.3,
                    "return_full_text": False,   # only return the new generated part
                    "do_sample": True,
                },
                "options": {
                    "wait_for_model": True,      # wait if model is loading (cold start)
                    "use_cache": False,
                }
            },
            timeout=90,
        )

        if resp.status_code == 503:
            log.warning("HuggingFace model is loading (cold start). Retrying once...")
            import time; time.sleep(20)
            resp = requests.post(HF_API,
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                json={"inputs": full_prompt,
                      "parameters": {"max_new_tokens": 900, "temperature": 0.3,
                                     "return_full_text": False, "do_sample": True},
                      "options": {"wait_for_model": True}},
                timeout=90)

        resp.raise_for_status()
        data = resp.json()

        # HF returns a list: [{"generated_text": "..."}]
        if isinstance(data, list) and data:
            text = data[0].get("generated_text", "").strip()
            # Strip any repeated prompt that leaked through
            if "[/INST]" in text:
                text = text.split("[/INST]")[-1].strip()
            return text if text else _fallback_message()

        return _fallback_message()

    except requests.exceptions.Timeout:
        log.error("HuggingFace API timed out.")
        return "⚠ The AI model timed out. The search results above are still valid — review the cases manually."
    except Exception as e:
        log.error(f"HuggingFace API error: {e}")
        return f"⚠ AI analysis unavailable ({e}). The search results above are still valid."


def _no_token_message() -> str:
    return (
        "ℹ AI analysis is disabled — no HuggingFace token set.\n\n"
        "To enable it:\n"
        "1. Sign up free at https://huggingface.co\n"
        "2. Go to Settings → Access Tokens → New Token (read access)\n"
        "3. Set environment variable:  HF_API_TOKEN=hf_xxxxxxxxxxxx\n"
        "4. Restart the server\n\n"
        "The search results above are fully functional without the AI token."
    )

def _fallback_message() -> str:
    return "⚠ AI model returned an empty response. The search results above are still valid."


# ── Main Analysis Function ────────────────────────────────────────────────────

def analyze_query(query: str, retrieved_cases: list[dict], full_texts: dict = None) -> dict:
    """
    Given a query + retrieved cases, generate AI legal analysis via HuggingFace.
    Returns dict with 'analysis', 'citations', 'cases_used'.
    """
    if not retrieved_cases:
        return {
            "analysis":   "No relevant cases found. Try different keywords or scrape more data.",
            "citations":  [],
            "cases_used": [],
        }

    # Build context block from retrieved cases
    case_blocks = []
    for i, case in enumerate(retrieved_cases[:5], 1):
        snippet = case.get("full_text", case.get("snippet", ""))[:1200]
        case_blocks.append(
            f"CASE {i}:\n"
            f"Court: {case.get('court_name', case.get('court', ''))}\n"
            f"Citation: {case.get('citation', 'N/A')}\n"
            f"Case No: {case.get('case_number', 'N/A')}\n"
            f"Title: {case.get('title', '')[:120]}\n"
            f"Date: {case.get('date', 'N/A')}\n"
            f"Relevance: {int(case.get('similarity', 0) * 100)}%\n"
            f"Excerpt:\n{snippet}\n"
        )

    context = "\n" + "─" * 50 + "\n" + "\n".join(case_blocks) + "─" * 50

    prompt = (
        f"A lawyer has this legal query about Pakistani law:\n\n"
        f"QUERY: {query}\n\n"
        f"I retrieved these {len(retrieved_cases)} most relevant cases from SHC, LHC, and IHC:\n"
        f"{context}\n\n"
        f"Provide a comprehensive legal analysis with citations."
    )

    analysis_text = ask_llm(prompt)

    # Extract citation patterns from the analysis text
    citations = re.findall(
        r"\d{4}\s+(?:SHC|LHC|IHC|PLD|SCMR|CLC|PLJ)\s*\w*\s*\d*",
        analysis_text
    )
    # Also include citations directly from retrieved cases
    for c in retrieved_cases:
        if c.get("citation") and c["citation"] not in citations:
            citations.append(c["citation"])

    return {
        "analysis":   analysis_text,
        "citations":  list(dict.fromkeys(citations)),  # deduplicate preserving order
        "cases_used": [
            {
                "citation":   c.get("citation"),
                "title":      c.get("title", "")[:100],
                "court":      c.get("court_name", c.get("court")),
                "url":        c.get("url"),
                "similarity": c.get("similarity"),
            }
            for c in retrieved_cases[:5]
        ],
    }


if __name__ == "__main__":
    # Quick test — set HF_API_TOKEN in your environment first
    logging.basicConfig(level=logging.INFO)
    test_cases = [{
        "court": "SHC", "court_name": "Sindh High Court",
        "citation": "2019 SHC KHI 1456",
        "title": "Muhammad Akram v. NHA",
        "similarity": 0.92,
        "full_text": "NHA held liable for road accident due to missing U-turn signs and road markings.",
        "id": "test1",
    }]
    result = analyze_query("NHA road accident missing signs", test_cases)
    print("\n=== Analysis ===")
    print(result["analysis"][:600])
    print("\n=== Citations ===")
    print(result["citations"])
