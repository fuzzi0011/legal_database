"""
Legal AI Engine — Production Ready
Always gives a response. Two permanently-online free providers:

  PRIMARY:  Groq (Llama 3.1 8B) — free, no CC, 300+ tok/s, always fast
  FALLBACK: Google Gemini Flash  — free, no CC, 1500 req/day

Get keys (both free, both take 2 minutes, no credit card):
  Groq:   https://console.groq.com → API Keys → Create Key
  Gemini: https://aistudio.google.com → Get API Key

Set env vars:
  GROQ_API_KEY   = gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  GEMINI_API_KEY = AIzaSy_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"""

import os, re, logging, requests

log = logging.getLogger(__name__)

GROQ_KEY   = os.getenv("GROQ_API_KEY",   "")
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")

LEGAL_SYSTEM = """You are a senior Pakistani legal research assistant with expertise in 
case law from the Sindh High Court (SHC), Lahore High Court (LHC), and Islamabad High Court (IHC).

When given a legal query and relevant case excerpts, provide a structured analysis:

**Legal Issue:** One-line summary of the legal question
**Relevant Cases:** Most applicable cases with full citations
**Court Decisions:** What the courts decided and their legal reasoning  
**Legal Principles:** Key rules and principles established by the courts
**Analysis:** How these precedents apply to the current situation
**Citations:** Complete list of all citations

Always cite: court name, citation number, year. Be precise and analytical.
This is legal research assistance, not legal advice."""


# ── Provider 1: GROQ ─────────────────────────────────────────────────────────

def _ask_groq(messages: list) -> str:
    if not GROQ_KEY:
        return ""
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": messages,
                "max_tokens": 900,
                "temperature": 0.3,
            },
            timeout=30,
        )
        if resp.status_code == 200:
            text = resp.json()["choices"][0]["message"]["content"].strip()
            log.info(f"[Groq] OK — {len(text)} chars")
            return text
        if resp.status_code == 429:
            # Rate limited — try different model
            resp2 = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
                json={"model": "gemma2-9b-it", "messages": messages,
                      "max_tokens": 900, "temperature": 0.3},
                timeout=30,
            )
            if resp2.status_code == 200:
                text = resp2.json()["choices"][0]["message"]["content"].strip()
                log.info(f"[Groq/gemma2] OK — {len(text)} chars")
                return text
        log.warning(f"[Groq] {resp.status_code}: {resp.text[:150]}")
    except Exception as e:
        log.warning(f"[Groq] Error: {e}")
    return ""


# ── Provider 2: Google Gemini Flash ──────────────────────────────────────────

def _ask_gemini(prompt: str) -> str:
    if not GEMINI_KEY:
        return ""
    try:
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.0-flash:generateContent?key={GEMINI_KEY}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 900, "temperature": 0.3},
            },
            timeout=30,
        )
        if resp.status_code == 200:
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            log.info(f"[Gemini] OK — {len(text)} chars")
            return text
        log.warning(f"[Gemini] {resp.status_code}: {resp.text[:150]}")
    except Exception as e:
        log.warning(f"[Gemini] Error: {e}")
    return ""


# ── Master function ───────────────────────────────────────────────────────────

def ask_llm(query: str, cases: list[dict] = None) -> str:
    """
    Send query + cases to LLM. Tries Groq first, Gemini as backup.
    Always returns a response — either AI analysis or setup instructions.
    """
    if not GROQ_KEY and not GEMINI_KEY:
        return _setup_message()

    # Build case context
    case_text = ""
    if cases:
        blocks = []
        for i, c in enumerate(cases[:5], 1):
            snippet = c.get("full_text", c.get("snippet", ""))[:1000]
            blocks.append(
                f"CASE {i}:\n"
                f"Court: {c.get('court_name', c.get('court',''))}\n"
                f"Citation: {c.get('citation','N/A')}\n"
                f"Title: {c.get('title','')[:100]}\n"
                f"Date: {c.get('date','N/A')}\n"
                f"Excerpt:\n{snippet}"
            )
        case_text = "\n\n".join(blocks)

    user_msg = (
        f"Retrieved cases from Pakistani court database:\n\n{case_text}\n\n"
        f"Legal Query: {query}\n\n"
        f"Provide a comprehensive legal analysis with Pakistani court citations."
        if case_text else
        f"Legal Query: {query}"
    )

    messages = [
        {"role": "system", "content": LEGAL_SYSTEM},
        {"role": "user",   "content": user_msg},
    ]
    full_prompt = f"{LEGAL_SYSTEM}\n\n{user_msg}"

    # Try Groq first
    if GROQ_KEY:
        result = _ask_groq(messages)
        if result:
            return result

    # Try Gemini
    if GEMINI_KEY:
        result = _ask_gemini(full_prompt)
        if result:
            return result

    return (
        "⚠ Both AI providers are temporarily unavailable right now. "
        "Please try again in 30 seconds."
    )


def _setup_message() -> str:
    return """ℹ️  No AI provider configured. Set at least ONE free API key:

OPTION 1 — GROQ (Recommended, fastest)
  • Sign up free: https://console.groq.com
  • Click: API Keys → Create API Key (no credit card)
  • Windows:   set GROQ_API_KEY=gsk_xxxxx
  • Mac/Linux: export GROQ_API_KEY=gsk_xxxxx

OPTION 2 — Google Gemini (1500 req/day free)
  • Sign up: https://aistudio.google.com (just Google account)
  • Click: Get API Key (no credit card)
  • Windows:   set GEMINI_API_KEY=AIzaSy_xxxxx
  • Mac/Linux: export GEMINI_API_KEY=AIzaSy_xxxxx

Restart the server after setting the key."""


# ── Called by FastAPI server ──────────────────────────────────────────────────

def analyze_query(query: str, retrieved_cases: list[dict], full_texts: dict = None) -> dict:
    if not retrieved_cases:
        return {
            "analysis":   "No relevant cases found. Try different keywords or run the scraper first.",
            "citations":  [],
            "cases_used": [],
        }

    analysis_text = ask_llm(query, retrieved_cases)

    # Extract citations from response text
    citations = re.findall(
        r"\d{4}\s+(?:SHC|LHC|IHC|PLD|SCMR|CLC|PLJ|MLD|NLR)\s*\w*\s*\d*",
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
                "title":      c.get("title","")[:100],
                "court":      c.get("court_name", c.get("court")),
                "url":        c.get("url"),
                "similarity": c.get("similarity"),
            }
            for c in retrieved_cases[:5]
        ],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"GROQ_API_KEY  : {'SET ✅' if GROQ_KEY   else 'NOT SET ❌'}")
    print(f"GEMINI_API_KEY: {'SET ✅' if GEMINI_KEY else 'NOT SET ❌'}")

    if not GROQ_KEY and not GEMINI_KEY:
        print("\n" + _setup_message())
    else:
        sample = [{
            "court": "SHC", "court_name": "Sindh High Court",
            "citation": "2019 SHC KHI 1456",
            "title": "Muhammad Akram v. National Highway Authority",
            "date": "2019-03-14", "similarity": 0.92,
            "full_text": (
                "NHA held liable for road accident due to absence of road markings "
                "and missing U-turn signs. Compensation Rs 3,500,000 awarded. "
                "Section 13 National Highways Act 1991 — non-delegable duty of care."
            ),
        }]
        r = analyze_query("NHA road accident missing signs liability", sample)
        print("\n=== Analysis ===\n", r["analysis"])
        print("\n=== Citations ===\n", r["citations"])
