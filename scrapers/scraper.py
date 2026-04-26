import requests
from bs4 import BeautifulSoup
import time

HEADERS = {"User-Agent": "Mozilla/5.0"}


def safe_get(url, params=None):
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=15)
        r.raise_for_status()
        time.sleep(0.2)
        return r
    except:
        return None


# ───────── LHC ─────────
class LHCScraper:
    BASE = "https://www.lhc.gov.pk"

    def fetch_all(self, max_pages=10):
        results = []

        for page in range(1, max_pages + 1):
            resp = safe_get(f"{self.BASE}/judgments", {"page": page})
            if not resp:
                break

            soup = BeautifulSoup(resp.text, "lxml")
            rows = soup.select("table tr")

            if not rows:
                break

            for r in rows[1:]:
                cols = r.find_all("td")
                if len(cols) < 3:
                    continue

                link = cols[0].find("a")
                if not link:
                    continue

                results.append({
                    "court": "LHC",
                    "title": link.text.strip(),
                    "url": self.BASE + link.get("href"),
                    "date": cols[1].text.strip(),
                    "citation": cols[2].text.strip(),
                    "full_text": ""  # disabled for now
                })

        return results


# ───────── SHC ─────────
class SHCScraper:
    BASE = "https://caselaw.shc.gov.pk"

    def fetch_all(self, max_pages=10):
        results = []

        for page in range(1, max_pages + 1):
            resp = safe_get(f"{self.BASE}/caselaw/search", {"page": page})
            if not resp:
                break

            soup = BeautifulSoup(resp.text, "lxml")
            cards = soup.select(".case")

            if not cards:
                break

            for c in cards:
                link = c.find("a")
                if not link:
                    continue

                results.append({
                    "court": "SHC",
                    "title": link.text.strip(),
                    "url": self.BASE + link.get("href"),
                    "date": "",
                    "citation": "",
                    "full_text": ""
                })

        return results


# ───────── IHC ─────────
class IHCScraper:
    BASE = "https://ihc.gov.pk"

    def fetch_all(self, max_pages=10):
        results = []

        for page in range(1, max_pages + 1):
            resp = safe_get(f"{self.BASE}/judgments", {"page": page})
            if not resp:
                break

            soup = BeautifulSoup(resp.text, "lxml")
            rows = soup.select("table tr")

            if not rows:
                break

            for r in rows[1:]:
                cols = r.find_all("td")
                if len(cols) < 2:
                    continue

                link = cols[0].find("a")
                if not link:
                    continue

                results.append({
                    "court": "IHC",
                    "title": link.text.strip(),
                    "url": self.BASE + link.get("href"),
                    "date": cols[1].text.strip(),
                    "citation": "",
                    "full_text": ""
                })

        return results
