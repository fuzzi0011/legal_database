"""
Legal AI Engine — HuggingFace Chat Completions (2025 API)
Uses the new HuggingFace router with OpenAI-compatible chat endpoint.
Free model: meta-llama/Meta-Llama-3.1-8B-Instruct (no credit card needed)

Get your free token: https://huggingface.co/settings/tokens
Set env var: HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
"""

import os
import re
import logging
import requests

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_API_TOKEN", "")

# New HuggingFace router URL (2025) — OpenAI-compatible chat completions
HF_ROUTER_URL = "https://router.huggingface.co/v1"

# Best free models available on HuggingFace router (try in order)
HF_MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",   # Best quality, free
    "mistralai/Mistral-7B-Instruct-v0.3",        # Fallback option
    "HuggingFaceH4/zephyr-7b-beta",              # Another fallback
]

SYSTEM_PROMPT = """You are a senior Pakistani legal research assistant specializing in case law 
from the Sindh High Court (SHC), Lahore High Court (LHC), and Islamabad High Court (IHC).

Given a legal query and retrieved case excerpts, provide a structured legal analysis:

**Legal Issue**: One line summary of the legal question
**Relevant Cases**: Most applicable cases with citations
**Court Decisions**: What courts decided and the reasoning
**Legal Principles**: Key legal rules and principles established  
**Analysis**: How precedents apply to the current situation
**Citations**: Full citation list

Always cite: court name, citation number, year, and judge name if available.
Be precise and analytical. Note: This is research assistance, not legal advice."""


def ask_llm(prompt: str) -> str:
    """
    Send prompt to HuggingFace via new router API (chat completions).
    Tries multiple free models in sequence.
    """
    if not HF_TOKEN:
        return _no_token_message()

    for model in HF_MODELS:
        result = _try_model(model, prompt)
        if result and not result.startswith("⚠"):
            return result

    return "⚠ All HuggingFace models unavailable right now. Try again in a minute. Search results above are still valid."


def _try_model(model: str, prompt: str) -> str:
    """Try one model via HF router chat completions endpoint."""
    try:
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            "max_tokens": 900,
            "temperature": 0.3,
            "stream": False,
        }

        resp = requests.post(
            f"{HF_ROUTER_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=90,
        )

        if resp.status_code == 503:
            log.warning(f"[{model}] Model loading (503), waiting 15s...")
            import time; time.sleep(15)
            resp = requests.post(
                f"{HF_ROUTER_URL}/chat/completions",
                headers=headers, json=payload, timeout=90
            )

        if resp.status_code == 404:
            log.warning(f"[{model}] Not found on router, trying next model.")
            return f"⚠ Model {model} not available."

        if resp.status_code == 401:
            return "⚠ Invalid HuggingFace token. Check your HF_API_TOKEN."

        resp.raise_for_status()
        data = resp.json()

        # Standard OpenAI-compatible response format
        text = data["choices"][0]["message"]["content"].strip()
        log.info(f"[{model}] Got response, length={len(text)}")
        return text

    except requests.exceptions.Timeout:
        log.warning(f"[{model}] Timed out.")
        return f"⚠ {model} timed out."
    except Exception as e:
        log.warning(f"[{model}] Error: {e}")
        return f"⚠ {model} error: {e}"


def _no_token_message() -> str:
    return (
        "ℹ️  AI analysis disabled — HuggingFace token not set.\n\n"
        "To enable free AI analysis:\n"
        "  1. Sign up free at https://huggingface.co\n"
        "  2. Go to Settings → Access Tokens → New Token (Read access)\n"
        "  3. Set it before starting the server:\n"
        "       Windows:   set HF_API_TOKEN=hf_xxxxxxxxxx\n"
        "       Mac/Linux: export HF_API_TOKEN=hf_xxxxxxxxxx\n"
        "  4. Restart the server\n\n"
        "Search results above are fully functional without the token."
    )


# ── Main Analysis Function ────────────────────────────────────────────────────

def analyze_query(query: str, retrieved_cases: list[dict], full_texts: dict = None) -> dict:
    """
    Given a query + retrieved cases → generate AI legal analysis via HuggingFace.
    Returns dict with 'analysis', 'citations', 'cases_used'.
    """
    if not retrieved_cases:
        return {
            "analysis":   "No relevant cases found. Try different keywords or scrape more data.",
            "citations":  [],
            "cases_used": [],
        }

    # Build context from retrieved cases
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

    context = "\n" + "─"*50 + "\n" + "\n".join(case_blocks) + "─"*50

    prompt = (
        f"A Pakistani lawyer has this legal research query:\n\n"
        f"QUERY: {query}\n\n"
        f"I retrieved these {len(retrieved_cases)} most relevant cases from SHC, LHC, and IHC databases:\n"
        f"{context}\n\n"
        f"Provide a comprehensive legal analysis with citations."
    )

    analysis_text = ask_llm(prompt)

    # Extract citation patterns
    citations = re.findall(
        r"\d{4}\s+(?:SHC|LHC|IHC|PLD|SCMR|CLC|PLJ)\s*\w*\s*\d*",
        analysis_text
    )
    for c in retrieved_cases:
        if c.get("citation") and c["citation"] not in citations:
            citations.append(c["citation"])

    return {
        "analysis":   analysis_text,
        "citations":  list(dict.fromkeys(citations)),
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
    logging.basicConfig(level=logging.INFO)
    # Quick connection test
    if not HF_TOKEN:
        print("Set HF_API_TOKEN env variable first.")
    else:
        print(f"Testing with token: {HF_TOKEN[:8]}...")
        result = ask_llm("What is the NHA's duty of care on national highways in Pakistan?")
        print(result[:500])
