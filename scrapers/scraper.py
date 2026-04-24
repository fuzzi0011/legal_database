"""
Pakistan High Courts - Public Judgment Scraper
Scrapes SHC (Sindh), LHC (Lahore), IHC (Islamabad) public case law portals.
All data scraped is publicly available on official government court websites.
"""

import requests
from bs4 import BeautifulSoup
import json
import os
import time
import random
import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlencode
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}


def get(url, params=None, retries=3, delay=2):
    """Polite HTTP GET with retries and rate limiting."""
    for attempt in range(retries):
        try:
            time.sleep(delay + random.uniform(0.5, 1.5))
            resp = requests.get(url, params=params, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            return resp
        except Exception as e:
            log.warning(f"Attempt {attempt+1} failed for {url}: {e}")
            time.sleep(delay * (attempt + 1))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SINDH HIGH COURT  (caselaw.shc.gov.pk)
# ─────────────────────────────────────────────────────────────────────────────

class SHCScraper:
    BASE = "https://caselaw.shc.gov.pk"
    SEARCH_URL = f"{BASE}/caselaw/search-all/search"
    HOME_URL   = f"{BASE}/caselaw/public/home"

    def search(self, keyword: str, max_pages: int = 5) -> list[dict]:
        """Search SHC for a keyword, paginate, return list of case metadata."""
        results = []
        log.info(f"[SHC] Searching: '{keyword}'")

        for page in range(1, max_pages + 1):
            params = {
                "keywords": keyword,
                "page": page,
                "perPage": 20,
            }
            resp = get(self.SEARCH_URL, params=params)
            if not resp:
                break

            soup = BeautifulSoup(resp.text, "lxml")
            # SHC lists judgments in card/row format
            cards = soup.select(".judgment-item, .case-row, tr.odd, tr.even, .card")
            if not cards:
                # fallback: try to find any links with case numbers
                cards = soup.find_all("a", href=re.compile(r"judgment|case|order", re.I))

            if not cards:
                log.info(f"[SHC] No more results at page {page}")
                break

            for card in cards:
                meta = self._parse_card(card)
                if meta:
                    results.append(meta)

            log.info(f"[SHC] Page {page}: {len(cards)} items found")

        return results

    def fetch_recent(self, max_pages: int = 10) -> list[dict]:
        """Fetch recent judgments from SHC home listing."""
        results = []
        log.info("[SHC] Fetching recent judgments from home page")

        resp = get(self.HOME_URL)
        if not resp:
            return results

        soup = BeautifulSoup(resp.text, "lxml")
        # Parse the home listing - SHC shows recent judgments as a list
        items = soup.select("table tr, .judgment-row, li.judgment")

        for item in items:
            meta = self._parse_card(item)
            if meta:
                results.append(meta)

        log.info(f"[SHC] Found {len(results)} recent judgments")
        return results

    def fetch_judgment_text(self, url: str) -> str:
        """Fetch full text of a single judgment."""
        resp = get(url)
        if not resp:
            return ""
        soup = BeautifulSoup(resp.text, "lxml")
        # Try common content containers
        for selector in [".judgment-text", "#judgment-content", ".case-content",
                          "div.content", "div#content", ".order-text", "article"]:
            el = soup.select_one(selector)
            if el:
                return el.get_text(separator="\n", strip=True)
        # Fallback: get all paragraph text
        paras = soup.find_all("p")
        return "\n".join(p.get_text(strip=True) for p in paras if len(p.get_text()) > 30)

    def _parse_card(self, el) -> dict | None:
        try:
            text = el.get_text(" ", strip=True)
            link = el.find("a")
            href = ""
            if link and link.get("href"):
                href = urljoin(self.BASE, link["href"])

            # Extract citation pattern like "2024 SHC KHI 123"
            citation = ""
            m = re.search(r"\d{4}\s+SHC\s+\w+\s+\d+", text)
            if m:
                citation = m.group()

            # Extract case number pattern
            case_no = ""
            m2 = re.search(r"(Const\.|C\.P\.|Crl\.|C\.A\.|R\.A\.|W\.P\.)\s*\w*\.?\s*\d+/\d{4}", text)
            if m2:
                case_no = m2.group()

            if not text or len(text) < 10:
                return None

            return {
                "court": "SHC",
                "court_name": "Sindh High Court",
                "citation": citation,
                "case_number": case_no,
                "title": text[:200],
                "url": href,
                "scraped_at": datetime.now().isoformat(),
                "full_text": "",
            }
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# LAHORE HIGH COURT  (lhc.gov.pk)
# ─────────────────────────────────────────────────────────────────────────────

class LHCScraper:
    BASE = "https://www.lhc.gov.pk"
    JUDGMENT_URL = f"{BASE}/judgments"
    SEARCH_URL   = f"{BASE}/searchjudgment"

    def search(self, keyword: str, max_pages: int = 5) -> list[dict]:
        """Search LHC judgment database."""
        results = []
        log.info(f"[LHC] Searching: '{keyword}'")

        for page in range(1, max_pages + 1):
            params = {"search": keyword, "page": page}
            resp = get(self.SEARCH_URL, params=params)
            if not resp:
                break

            soup = BeautifulSoup(resp.text, "lxml")
            rows = soup.select("table tbody tr, .judgment-row, .search-result-item")
            if not rows:
                break

            for row in rows:
                meta = self._parse_row(row)
                if meta:
                    results.append(meta)

            log.info(f"[LHC] Page {page}: {len(rows)} rows")

        return results

    def fetch_recent(self, max_pages: int = 10) -> list[dict]:
        """Fetch recent LHC judgments."""
        results = []
        log.info("[LHC] Fetching recent judgments")

        for page in range(1, max_pages + 1):
            resp = get(self.JUDGMENT_URL, params={"page": page})
            if not resp:
                break

            soup = BeautifulSoup(resp.text, "lxml")
            rows = soup.select("table tbody tr, .judgment-item")
            if not rows:
                break

            for row in rows:
                meta = self._parse_row(row)
                if meta:
                    results.append(meta)

            log.info(f"[LHC] Page {page}: {len(rows)} rows")

        return results

    def fetch_judgment_text(self, url: str) -> str:
        resp = get(url)
        if not resp:
            return ""
        soup = BeautifulSoup(resp.text, "lxml")
        for sel in [".judgment-body", "#judgment", ".order-text", "div.content", "article"]:
            el = soup.select_one(sel)
            if el:
                return el.get_text("\n", strip=True)
        paras = soup.find_all("p")
        return "\n".join(p.get_text(strip=True) for p in paras if len(p.get_text()) > 30)

    def _parse_row(self, el) -> dict | None:
        try:
            cells = el.find_all(["td", "li", "div"])
            text  = el.get_text(" ", strip=True)
            link  = el.find("a")
            href  = urljoin(self.BASE, link["href"]) if link and link.get("href") else ""

            citation = ""
            m = re.search(r"\d{4}\s+(LHC|PLD|SCMR|CLC)\s+\w*\s*\d*", text)
            if m:
                citation = m.group()

            date_str = ""
            m2 = re.search(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", text)
            if m2:
                date_str = m2.group()

            if len(text) < 10:
                return None

            return {
                "court": "LHC",
                "court_name": "Lahore High Court",
                "citation": citation,
                "case_number": "",
                "title": text[:200],
                "url": href,
                "date": date_str,
                "scraped_at": datetime.now().isoformat(),
                "full_text": "",
            }
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# ISLAMABAD HIGH COURT  (ihc.gov.pk / mis.ihc.gov.pk)
# ─────────────────────────────────────────────────────────────────────────────

class IHCScraper:
    BASE       = "https://www.ihc.gov.pk"
    MIS_BASE   = "https://mis.ihc.gov.pk"
    JUDGMENT_URL = f"{BASE}/judgments"
    ALT_URL      = f"{MIS_BASE}/frmJudgmentSearch"

    def search(self, keyword: str, max_pages: int = 5) -> list[dict]:
        results = []
        log.info(f"[IHC] Searching: '{keyword}'")

        for url in [self.JUDGMENT_URL, self.ALT_URL]:
            resp = get(url, params={"search": keyword})
            if not resp:
                continue
            soup = BeautifulSoup(resp.text, "lxml")
            rows = soup.select("table tbody tr, .judgment-row, .result-item")
            for row in rows:
                meta = self._parse_row(row)
                if meta:
                    results.append(meta)
            if results:
                log.info(f"[IHC] Found {len(results)} via {url}")
                break

        return results

    def fetch_recent(self, max_pages: int = 5) -> list[dict]:
        results = []
        log.info("[IHC] Fetching recent IHC judgments")

        for page in range(1, max_pages + 1):
            resp = get(self.JUDGMENT_URL, params={"page": page})
            if not resp:
                break
            soup = BeautifulSoup(resp.text, "lxml")
            rows = soup.select("table tbody tr, .judgment-item, li.judgment")
            if not rows:
                break
            for row in rows:
                meta = self._parse_row(row)
                if meta:
                    results.append(meta)
            log.info(f"[IHC] Page {page}: {len(rows)} rows")

        return results

    def fetch_judgment_text(self, url: str) -> str:
        resp = get(url)
        if not resp:
            return ""
        soup = BeautifulSoup(resp.text, "lxml")
        for sel in [".judgment-body", "#order-text", ".content-area", "div.content", "article"]:
            el = soup.select_one(sel)
            if el:
                return el.get_text("\n", strip=True)
        paras = soup.find_all("p")
        return "\n".join(p.get_text(strip=True) for p in paras if len(p.get_text()) > 30)

    def _parse_row(self, el) -> dict | None:
        try:
            text = el.get_text(" ", strip=True)
            link = el.find("a")
            href = urljoin(self.BASE, link["href"]) if link and link.get("href") else ""

            citation = ""
            m = re.search(r"\d{4}\s+(IHC|PLD|SCMR)\s+\w*\s*\d*", text)
            if m:
                citation = m.group()

            if len(text) < 10:
                return None

            return {
                "court": "IHC",
                "court_name": "Islamabad High Court",
                "citation": citation,
                "case_number": "",
                "title": text[:200],
                "url": href,
                "scraped_at": datetime.now().isoformat(),
                "full_text": "",
            }
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def scrape_all(keyword: str = None, fetch_full_text: bool = True,
               max_pages: int = 5, save: bool = True) -> list[dict]:
    """
    Run all three scrapers. Optionally fetch full text for each case.
    Returns combined list of case dicts.
    """
    scrapers = [
        ("SHC", SHCScraper()),
        ("LHC", LHCScraper()),
        ("IHC", IHCScraper()),
    ]

    all_cases = []

    for court_code, scraper in scrapers:
        try:
            if keyword:
                cases = scraper.search(keyword, max_pages=max_pages)
            else:
                cases = scraper.fetch_recent(max_pages=max_pages)

            # Fetch full text for each case (politely)
            if fetch_full_text:
                for i, case in enumerate(cases):
                    if case.get("url"):
                        log.info(f"[{court_code}] Fetching full text {i+1}/{len(cases)}: {case['url']}")
                        case["full_text"] = scraper.fetch_judgment_text(case["url"])

            all_cases.extend(cases)
            log.info(f"[{court_code}] Total: {len(cases)} cases")

        except Exception as e:
            log.error(f"[{court_code}] Scraper error: {e}")

    if save and all_cases:
        out_file = DATA_DIR / f"cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_cases, f, ensure_ascii=False, indent=2)
        log.info(f"Saved {len(all_cases)} cases to {out_file}")

    return all_cases


if __name__ == "__main__":
    import sys
    kw = sys.argv[1] if len(sys.argv) > 1 else None
    cases = scrape_all(keyword=kw, fetch_full_text=True, max_pages=3)
    print(f"\n✅ Scraped {len(cases)} total cases")
