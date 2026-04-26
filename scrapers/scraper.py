import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import time
import logging

log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0"}

def safe_get(url, params=None):
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=15)
        r.raise_for_status()
        time.sleep(0.2)
        return r
    except Exception as e:
        log.error(f"Request failed: {url} — {e}")
        return None


# ─────────────────────────────────────────────
# LHC SCRAPER
# ─────────────────────────────────────────────

class LHCScraper:
    BASE = "https://www.lhc.gov.pk"

    def fetch_all(self, max_pages=500):
        results = []

        for page in range(1, max_pages + 1):
            url = f"{self.BASE}/judgments"
            resp = safe_get(url, params={"page": page})
            if not resp:
                break

            soup = BeautifulSoup(resp.text, "lxml")
            rows = soup.select("table tr")

            if not rows:
                break

            page_cases = []

            for r in rows[1:]:
                cols = r.find_all("td")
                if len(cols) < 3:
                    continue

                link = cols[0].find("a")
                if not link:
                    continue

                case = {
                    "court": "LHC",
                    "title": link.text.strip(),
                    "url": self.BASE + link.get("href"),
                    "date": cols[1].text.strip(),
                    "citation": cols[2].text.strip(),
                }
                page_cases.append(case)

            if not page_cases:
                break

            log.info(f"LHC page {page}: {len(page_cases)} cases")
            results.extend(page_cases)

        return results

    def fetch_judgment_text(self, url):
        resp = safe_get(url)
        if not resp:
            return ""

        soup = BeautifulSoup(resp.text, "lxml")
        return soup.get_text(" ", strip=True)[:20000]


# ─────────────────────────────────────────────
# SHC SCRAPER
# ─────────────────────────────────────────────

class SHCScraper:
    BASE = "https://caselaw.shc.gov.pk"

    def fetch_all(self, max_pages=300):
        results = []

        for page in range(1, max_pages + 1):
            url = f"{self.BASE}/caselaw/search"
            resp = safe_get(url, params={"page": page})
            if not resp:
                break

            soup = BeautifulSoup(resp.text, "lxml")
            cards = soup.select(".case")

            if not cards:
                break

            page_cases = []

            for c in cards:
                link = c.find("a")
                if not link:
                    continue

                case = {
                    "court": "SHC",
                    "title": link.text.strip(),
                    "url": self.BASE + link.get("href"),
                    "date": "",
                    "citation": "",
                }
                page_cases.append(case)

            log.info(f"SHC page {page}: {len(page_cases)} cases")
            results.extend(page_cases)

        return results

    def fetch_judgment_text(self, url):
        resp = safe_get(url)
        if not resp:
            return ""

        soup = BeautifulSoup(resp.text, "lxml")
        return soup.get_text(" ", strip=True)[:20000]


# ─────────────────────────────────────────────
# IHC SCRAPER
# ─────────────────────────────────────────────

class IHCScraper:
    BASE = "https://ihc.gov.pk"

    def fetch_all(self, max_pages=300):
        results = []

        for page in range(1, max_pages + 1):
            url = f"{self.BASE}/judgments"
            resp = safe_get(url, params={"page": page})
            if not resp:
                break

            soup = BeautifulSoup(resp.text, "lxml")
            rows = soup.select("table tr")

            if not rows:
                break

            page_cases = []

            for r in rows[1:]:
                cols = r.find_all("td")
                if len(cols) < 2:
                    continue

                link = cols[0].find("a")
                if not link:
                    continue

                case = {
                    "court": "IHC",
                    "title": link.text.strip(),
                    "url": self.BASE + link.get("href"),
                    "date": cols[1].text.strip(),
                    "citation": "",
                }
                page_cases.append(case)

            log.info(f"IHC page {page}: {len(page_cases)} cases")
            results.extend(page_cases)

        return results

    def fetch_judgment_text(self, url):
        resp = safe_get(url)
        if not resp:
            return ""

        soup = BeautifulSoup(resp.text, "lxml")
        return soup.get_text(" ", strip=True)[:20000]


# ─────────────────────────────────────────────
# MULTI-THREAD ENRICHMENT
# ─────────────────────────────────────────────

def enrich_cases(scraper, cases):
    def process(case):
        if case.get("url"):
            case["full_text"] = scraper.fetch_judgment_text(case["url"])
        return case

    with ThreadPoolExecutor(max_workers=10) as executor:
        return list(executor.map(process, cases))
