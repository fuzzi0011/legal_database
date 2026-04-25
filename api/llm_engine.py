"""
Legal AI Engine — HuggingFace Inference Providers (Always Online)
Uses the correct 2025 HF API endpoint: /v1/chat/completions
This is permanently deployed — no Colab, no local server needed.

Model: mistralai/Mistral-7B-Instruct-v0.2  (same one that passed your tests)

Setup (one time):
  1. Go to https://huggingface.co/settings/tokens
  2. New Token → Token Type: READ → Generate
  3. Set env var: HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"""

import os, re, logging, requests

log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_API_TOKEN", "")

# Correct 2025 HuggingFace Inference Providers endpoint
# This is the OpenAI-compatible chat completions API — always online
HF_API_URL = "https://api-inference.huggingface.co/v1/chat/completions"

# Same model that passed all 3 tests in your Colab
PRIMARY_MODEL  = "mistralai/Mistral-7B-Instruct-v0.2"

# Fallback models (tried in order if primary fails)
FALLBACK_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceH4/zephyr-7b-beta",
    "microsoft/Phi-3-mini-4k-instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]

LEGAL_SYSTEM = """You are a senior Pakistani legal research assistant specializing in 
case law from the Sindh High Court (SHC), Lahore High Court (LHC), and Islamabad High Court (IHC).

When given a legal query, provide:
1. Legal Issue: one-line summary
2. Relevant Legal Principles: key rules that apply
3. Court Decisions: what Pakistani courts typically decide
4. Analysis: practical advice for the lawyer
5. Citations: any relevant Pakistani law or cases

Be precise. Cite Pakistani statutes and court citations where applicable."""


# ── Core API call ───────────────────────────────────────────────────────────

def _call_hf(model: str, messages: list, timeout: int = 60) -> str:
    """
    Call HuggingFace Inference Providers chat completions endpoint.
    Returns response text or empty string on failure.
    """
    resp = requests.post(
        HF_API_URL,
        headers={
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "max_tokens": 800,
            "temperature": 0.3,
            "stream": False,
        },
        timeout=timeout,
    )

    if resp.status_code == 200:
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    if resp.status_code == 401:
        raise ValueError("Invalid HuggingFace token. Check your HF_API_TOKEN.")

    if resp.status_code == 422:
        # Model not supported by this endpoint — try next
        log.warning(f"[HF] Model {model} returned 422 (unsupported), trying fallback.")
        return ""

    if resp.status_code == 503:
        log.warning(f"[HF] Model {model} is loading (503). Waiting 20s...")
        import time; time.sleep(20)
        # Retry once
        r2 = requests.post(
            HF_API_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "max_tokens": 800,
                  "temperature": 0.3, "stream": False},
            timeout=timeout,
        )
        if r2.status_code == 200:
            return r2.json()["choices"][0]["message"]["content"].strip()
        return ""

    if resp.status_code == 429:
        log.warning(f"[HF] Rate limit hit on {model}.")
        return ""

    log.warning(f"[HF] {model} returned {resp.status_code}: {resp.text[:200]}")
    return ""


def ask_llm(query: str, cases: list[dict] = None) -> str:
    """
    Send a legal query + retrieved cases to Mistral-7B via HuggingFace.
    Tries primary model, then fallbacks automatically.
    """
    if not HF_TOKEN:
        return _no_token_message()

    # Build the user message
    if cases:
        case_blocks = []
        for i, c in enumerate(cases[:5], 1):
            snippet = c.get("full_text", c.get("snippet", ""))[:1000]
            case_blocks.append(
                f"CASE {i}:\n"
                f"Court: {c.get('court_name', c.get('court', ''))}\n"
                f"Citation: {c.get('citation', 'N/A')}\n"
                f"Title: {c.get('title', '')[:100]}\n"
                f"Date: {c.get('date', 'N/A')}\n"
                f"Excerpt:\n{snippet}\n"
            )
        user_content = (
            f"I found these relevant cases from our legal database:\n\n"
            + "\n".join(case_blocks)
            + f"\n\nLegal Query: {query}\n\n"
            f"Based on the above cases, provide a comprehensive legal analysis "
            f"with proper Pakistani court citations."
        )
    else:
        user_content = f"Legal Query: {query}"

    messages = [
        {"role": "system", "content": LEGAL_SYSTEM},
        {"role": "user",   "content": user_content},
    ]

    # Try primary model
    try:
        log.info(f"[HF] Trying primary model: {PRIMARY_MODEL}")
        result = _call_hf(PRIMARY_MODEL, messages)
        if result:
            log.info(f"[HF] Primary model success — {len(result)} chars")
            return result
    except ValueError as e:
        # Invalid token — no point trying fallbacks
        return f"⚠ {e}"
    except requests.exceptions.Timeout:
        log.warning(f"[HF] Primary model timed out.")
    except Exception as e:
        log.warning(f"[HF] Primary model error: {e}")

    # Try fallback models
    for model in FALLBACK_MODELS:
        try:
            log.info(f"[HF] Trying fallback: {model}")
            result = _call_hf(model, messages)
            if result:
                log.info(f"[HF] Fallback {model} success — {len(result)} chars")
                return result
        except requests.exceptions.Timeout:
            log.warning(f"[HF] {model} timed out.")
        except Exception as e:
            log.warning(f"[HF] {model} error: {e}")

    return (
        "⚠ All models are temporarily unavailable. "
        "This usually means HuggingFace is under load — please try again in 1-2 minutes. "
        "Your search results above are still fully valid."
    )


def _no_token_message() -> str:
    return """ℹ️  AI analysis is disabled — HuggingFace token not set.

To enable Mistral-7B AI analysis (free, always online):

  1. Go to: https://huggingface.co/settings/tokens
  2. Click "New Token" → Token Type: READ → Generate
  3. Copy the token (starts with hf_...)
  4. Set it before starting the server:

     Windows:
       set HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
       python -m api.server

     Mac/Linux:
       export HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
       python -m api.server

     Railway (hosting):
       Add HF_API_TOKEN in Railway Dashboard → Variables tab

Search results above are fully functional without the token."""


# ── Main analysis function (called by FastAPI) ──────────────────────────────

def analyze_query(query: str, retrieved_cases: list[dict], full_texts: dict = None) -> dict:
    """
    Takes a legal query + retrieved cases → returns AI analysis with citations.
    This is called by api/server.py when user clicks "AI Analyze".
    """
    if not retrieved_cases:
        return {
            "analysis":   "No relevant cases found. Try different keywords or run the scraper.",
            "citations":  [],
            "cases_used": [],
        }

    analysis_text = ask_llm(query, retrieved_cases)

    # Extract citation patterns from response
    citations = re.findall(
        r"\d{4}\s+(?:SHC|LHC|IHC|PLD|SCMR|CLC|PLJ|MLD|NLR)\s*\w*\s*\d*",
        analysis_text
    )
    # Also include citations from the retrieved cases
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


# ── Quick test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print(f"HF Token set: {'YES ✅' if HF_TOKEN else 'NO ❌'}")
    print(f"Primary model: {PRIMARY_MODEL}")
    print(f"API endpoint: {HF_API_URL}")

    if not HF_TOKEN:
        print("\n" + _no_token_message())
    else:
        print("\nTesting with a sample legal query...")
        sample_cases = [{
            "court": "SHC",
            "court_name": "Sindh High Court",
            "citation": "2019 SHC KHI 1456",
            "title": "Muhammad Akram v. National Highway Authority & Others",
            "date": "2019-03-14",
            "similarity": 0.92,
            "full_text": (
                "NHA held liable for road accident due to absence of road markings, "
                "missing reflective signs, and no warning indicators near a sharp U-turn. "
                "Compensation of Rs. 3,500,000 awarded. Section 13 National Highways Act 1991."
            ),
        }]
        result = analyze_query("NHA road accident missing signs liability", sample_cases)
        print("\n=== Analysis ===")
        print(result["analysis"])
        print("\n=== Citations ===")
        print(result["citations"])
