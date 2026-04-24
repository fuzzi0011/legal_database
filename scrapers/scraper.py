"""
Pakistan High Courts — Mass Scraper
Covers ALL major legal fields across SHC, LHC, IHC.
Scrapes hundreds to thousands of cases depending on site availability.

Usage:
  python scrapers/scraper.py                    # scrape all fields, all courts
  python scrapers/scraper.py "NHA accident"     # specific keyword
  python scrapers/scraper.py --quick            # just recent judgments
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from bs4 import BeautifulSoup
import json, time, random, logging, re, argparse
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlencode, quote_plus

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── ALL legal fields to scrape across courts ──────────────────────────────────
ALL_LEGAL_FIELDS = [
    # Constitutional & Administrative
    "constitutional petition fundamental rights",
    "writ petition habeas corpus",
    "mandamus certiorari judicial review",
    "administrative law government action",
    "service matters government employees",
    "civil servants termination reinstatement",

    # Criminal Law
    "murder qatl-e-amd FIR conviction",
    "bail application criminal case",
    "drug trafficking CNSA narcotics",
    "robbery dacoity armed robbery sentence",
    "kidnapping abduction ransom",
    "cybercrime PECA electronic fraud",
    "corruption NAB accountability",
    "terrorism ATA anti terrorism",
    "rape sexual assault conviction",
    "blasphemy 295-C religion",

    # Family & Personal Law
    "divorce khula dissolution marriage",
    "custody children guardianship",
    "maintenance nafqa wife children",
    "inheritance succession property",
    "dowry mehr dower wife",
    "child marriage restraint act",
    "domestic violence protection",

    # Property & Land
    "property dispute ownership title",
    "illegal construction demolition LDA CDA",
    "land acquisition compensation",
    "tenancy eviction landlord tenant",
    "mutation revenue record",
    "benami transaction property",
    "mortgage foreclosure bank",
    "housing society fraud plot",

    # Contract & Commercial
    "breach of contract damages",
    "partnership dispute dissolution",
    "cheque dishonour section 489-F",
    "banking fraud loan default",
    "insurance claim repudiation",
    "arbitration award enforcement",
    "company law SECP liquidation",
    "intellectual property trademark copyright",

    # Road Accidents & NHA
    "NHA road accident negligence compensation",
    "motorway accident fatal injuries",
    "traffic accident death compensation",
    "reckless driving hit and run",
    "road safety signs markings missing",

    # Labour & Employment
    "wrongful termination labour court NIRC",
    "EOBI workers old age benefits",
    "minimum wage violation workers",
    "industrial relations strike lockout",
    "workmen compensation injury",
    "sexual harassment workplace PSHO",

    # Tax & Revenue
    "FBR tax evasion income tax",
    "sales tax fictitious invoices GST",
    "customs duty smuggling seizure",
    "property tax valuation appeal",
    "withholding tax deduction FBR",

    # Environment & Public Interest
    "environmental pollution EPA NEPA",
    "smog pollution Lahore air quality",
    "water contamination industrial effluent",
    "deforestation illegal cutting trees",
    "solid waste management municipal",

    # Education & Health
    "medical negligence malpractice hospital",
    "university admission merit list",
    "school fee private institutions",
    "PMDC medical college affiliation",
    "pharmacy drug quality DRAP",

    # Utility & Consumer
    "electricity NEPRA WAPDA overbilling",
    "gas OGRA price hike supply",
    "telephone PTCL consumer complaint",
    "water supply utility services",
    "price control consumer protection",

    # Civil Procedure
    "limitation period time barred suit",
    "res judicata estoppel",
    "specific performance contract property",
    "injunction temporary restraining order",
    "ex parte decree setting aside",
    "execution decree enforcement",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def polite_get(url, params=None, retries=3, delay=2.5):
    """Rate-limited HTTP GET with retries."""
    for attempt in range(retries):
        try:
            time.sleep(delay + random.uniform(0.5, 1.5))
            r = requests.get(url, params=params, headers=HEADERS, timeout=25)
            r.raise_for_status()
            return r
        except requests.exceptions.HTTPError as e:
            if r.status_code in (403, 429, 503):
                wait = delay * (attempt + 2) * 2
                log.warning(f"Rate limited ({r.status_code}), waiting {wait}s...")
                time.sleep(wait)
            else:
                log.warning(f"HTTP {r.status_code} for {url}")
                return None
        except Exception as e:
            log.warning(f"Attempt {attempt+1} failed: {e}")
            time.sleep(delay * (attempt + 1))
    return None


def extract_text_from_soup(soup: BeautifulSoup) -> str:
    """Extract judgment text from common page content containers."""
    for sel in [
        ".judgment-text", "#judgment-content", ".case-content",
        ".order-text", "#order-text", ".judgment-body",
        "div.content-area", "div#content", "article",
        "div.container div.row div.col", "div.panel-body",
    ]:
        el = soup.select_one(sel)
        if el and len(el.get_text()) > 200:
            return el.get_text("\n", strip=True)

    # Fallback: collect all substantial paragraphs
    paras = soup.find_all(["p", "div"], class_=False)
    texts = [p.get_text(strip=True) for p in paras if len(p.get_text(strip=True)) > 80]
    return "\n\n".join(texts[:50])


def parse_citation(text: str) -> str:
    patterns = [
        r"\d{4}\s+SHC\s+\w+\s+\d+",
        r"\d{4}\s+LHC\s+\d+",
        r"\d{4}\s+IHC\s+\d+",
        r"PLD\s+\d{4}\s+\w+\s+\d+",
        r"\d{4}\s+SCMR\s+\d+",
        r"\d{4}\s+CLC\s+\d+",
        r"\d{4}\s+MLD\s+\d+",
        r"\d{4}\s+PLJ\s+\w+\s+\d+",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group().strip()
    return ""


def parse_case_number(text: str) -> str:
    m = re.search(
        r"(Const\.|Crl\.|C\.P\.|W\.P\.|C\.A\.|R\.A\.|Civ\.|F\.A\.|"
        r"Cr\.A\.|H\.C\.A\.|R\.F\.A\.)\s*[\w.]*\s*No\.?\s*\d+/\d{4}",
        text, re.IGNORECASE
    )
    return m.group().strip() if m else ""


# ─────────────────────────────────────────────────────────────────────────────
# SINDH HIGH COURT
# ─────────────────────────────────────────────────────────────────────────────

class SHCScraper:
    BASE       = "https://caselaw.shc.gov.pk"
    SEARCH_URL = f"{BASE}/caselaw/search-all/search"
    LIST_URL   = f"{BASE}/caselaw/public/home"

    def search_keyword(self, keyword: str, max_pages: int = 5) -> list[dict]:
        results = []
        log.info(f"[SHC] Searching: '{keyword}'")
        for page in range(1, max_pages + 1):
            resp = polite_get(self.SEARCH_URL, params={
                "keywords": keyword, "page": page, "perPage": 25
            })
            if not resp:
                break
            soup = BeautifulSoup(resp.text, "lxml")
            items = self._extract_list_items(soup)
            if not items:
                break
            results.extend(items)
            log.info(f"[SHC] '{keyword}' page {page}: +{len(items)} cases")
        return results

    def fetch_recent(self, max_pages: int = 10) -> list[dict]:
        results = []
        log.info("[SHC] Fetching recent judgments")
        for page in range(1, max_pages + 1):
            resp = polite_get(self.LIST_URL, params={"page": page})
            if not resp:
                break
            soup = BeautifulSoup(resp.text, "lxml")
            items = self._extract_list_items(soup)
            if not items:
                break
            results.extend(items)
        return results

    def fetch_full_text(self, url: str) -> str:
        if not url or url == "#":
            return ""
        resp = polite_get(url)
        if not resp:
            return ""
        return extract_text_from_soup(BeautifulSoup(resp.text, "lxml"))

    def _extract_list_items(self, soup: BeautifulSoup) -> list[dict]:
        items = []
        # SHC uses table rows and result divs
        rows = soup.select("table tbody tr, .judgment-item, .result-row, .case-item")
        if not rows:
            rows = soup.find_all("a", href=re.compile(r"/case/|/judgment/|/order/", re.I))

        for row in rows:
            meta = self._parse_row(row)
            if meta:
                items.append(meta)
        return items

    def _parse_row(self, el) -> dict | None:
        try:
            text = el.get_text(" ", strip=True)
            if len(text) < 15:
                return None
            link = el.find("a") if el.name != "a" else el
            href = urljoin(self.BASE, link["href"]) if link and link.get("href") else ""
            return {
                "court": "SHC", "court_name": "Sindh High Court",
                "citation": parse_citation(text),
                "case_number": parse_case_number(text),
                "title": text[:250],
                "url": href,
                "date": self._extract_date(text),
                "full_text": "",
                "scraped_at": datetime.now().isoformat(),
            }
        except Exception:
            return None

    def _extract_date(self, text: str) -> str:
        m = re.search(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", text)
        return m.group() if m else ""


# ─────────────────────────────────────────────────────────────────────────────
# LAHORE HIGH COURT
# ─────────────────────────────────────────────────────────────────────────────

class LHCScraper:
    BASE        = "https://www.lhc.gov.pk"
    SEARCH_URL  = f"{BASE}/searchjudgment"
    LIST_URL    = f"{BASE}/judgments"

    def search_keyword(self, keyword: str, max_pages: int = 5) -> list[dict]:
        results = []
        log.info(f"[LHC] Searching: '{keyword}'")
        for page in range(1, max_pages + 1):
            resp = polite_get(self.SEARCH_URL, params={"search": keyword, "page": page})
            if not resp:
                break
            soup = BeautifulSoup(resp.text, "lxml")
            items = self._extract_list_items(soup)
            if not items:
                break
            results.extend(items)
            log.info(f"[LHC] '{keyword}' page {page}: +{len(items)} cases")
        return results

    def fetch_recent(self, max_pages: int = 10) -> list[dict]:
        results = []
        log.info("[LHC] Fetching recent judgments")
        for page in range(1, max_pages + 1):
            resp = polite_get(self.LIST_URL, params={"page": page})
            if not resp:
                break
            soup = BeautifulSoup(resp.text, "lxml")
            items = self._extract_list_items(soup)
            if not items:
                break
            results.extend(items)
        return results

    def fetch_full_text(self, url: str) -> str:
        if not url or url == "#":
            return ""
        resp = polite_get(url)
        if not resp:
            return ""
        return extract_text_from_soup(BeautifulSoup(resp.text, "lxml"))

    def _extract_list_items(self, soup: BeautifulSoup) -> list[dict]:
        items = []
        rows = soup.select("table tbody tr, .judgment-row, .search-result-item, .case-row")
        if not rows:
            rows = soup.find_all("a", href=re.compile(r"/judgment|/order|/case", re.I))
        for row in rows:
            meta = self._parse_row(row)
            if meta:
                items.append(meta)
        return items

    def _parse_row(self, el) -> dict | None:
        try:
            text = el.get_text(" ", strip=True)
            if len(text) < 15:
                return None
            link = el.find("a") if el.name != "a" else el
            href = urljoin(self.BASE, link["href"]) if link and link.get("href") else ""
            m_date = re.search(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", text)
            return {
                "court": "LHC", "court_name": "Lahore High Court",
                "citation": parse_citation(text),
                "case_number": parse_case_number(text),
                "title": text[:250],
                "url": href,
                "date": m_date.group() if m_date else "",
                "full_text": "",
                "scraped_at": datetime.now().isoformat(),
            }
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# ISLAMABAD HIGH COURT
# ─────────────────────────────────────────────────────────────────────────────

class IHCScraper:
    BASE      = "https://www.ihc.gov.pk"
    ALT_BASE  = "https://mis.ihc.gov.pk"
    LIST_URL  = f"{BASE}/judgments"
    ALT_SEARCH = f"{ALT_BASE}/frmJudgmentSearch"

    def search_keyword(self, keyword: str, max_pages: int = 5) -> list[dict]:
        results = []
        log.info(f"[IHC] Searching: '{keyword}'")
        # Try main site
        for page in range(1, max_pages + 1):
            resp = polite_get(self.LIST_URL, params={"search": keyword, "page": page})
            if not resp:
                break
            soup = BeautifulSoup(resp.text, "lxml")
            items = self._extract_list_items(soup, self.BASE)
            if not items:
                break
            results.extend(items)
            log.info(f"[IHC] '{keyword}' page {page}: +{len(items)} cases")

        # Also try MIS portal
        resp2 = polite_get(self.ALT_SEARCH, params={"keyword": keyword})
        if resp2:
            soup2 = BeautifulSoup(resp2.text, "lxml")
            items2 = self._extract_list_items(soup2, self.ALT_BASE)
            results.extend(items2)
            if items2:
                log.info(f"[IHC-MIS] +{len(items2)} additional cases")

        return results

    def fetch_recent(self, max_pages: int = 10) -> list[dict]:
        results = []
        log.info("[IHC] Fetching recent judgments")
        for page in range(1, max_pages + 1):
            resp = polite_get(self.LIST_URL, params={"page": page})
            if not resp:
                break
            soup = BeautifulSoup(resp.text, "lxml")
            items = self._extract_list_items(soup, self.BASE)
            if not items:
                break
            results.extend(items)
        return results

    def fetch_full_text(self, url: str) -> str:
        if not url or url == "#":
            return ""
        resp = polite_get(url)
        if not resp:
            return ""
        return extract_text_from_soup(BeautifulSoup(resp.text, "lxml"))

    def _extract_list_items(self, soup: BeautifulSoup, base: str) -> list[dict]:
        items = []
        rows = soup.select("table tbody tr, .judgment-row, .result-item, li.judgment")
        if not rows:
            rows = soup.find_all("a", href=re.compile(r"/judgment|/order|/case", re.I))
        for row in rows:
            meta = self._parse_row(row, base)
            if meta:
                items.append(meta)
        return items

    def _parse_row(self, el, base: str) -> dict | None:
        try:
            text = el.get_text(" ", strip=True)
            if len(text) < 15:
                return None
            link = el.find("a") if el.name != "a" else el
            href = urljoin(base, link["href"]) if link and link.get("href") else ""
            m_date = re.search(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", text)
            return {
                "court": "IHC", "court_name": "Islamabad High Court",
                "citation": parse_citation(text),
                "case_number": parse_case_number(text),
                "title": text[:250],
                "url": href,
                "date": m_date.group() if m_date else "",
                "full_text": "",
                "scraped_at": datetime.now().isoformat(),
            }
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# WORLDLII — free Pakistani case law index (no login required)
# ─────────────────────────────────────────────────────────────────────────────

class WorldLIIScraper:
    """
    WorldLII indexes thousands of Pakistani High Court cases publicly.
    URL: http://www.worldlii.org/pk/
    """
    BASE      = "http://www.worldlii.org"
    SEARCH_URL = "http://www.worldlii.org/cgi-bin/sinosrch.cgi"

    DATABASES = [
        "pk/cases/PKSC",   # Supreme Court
        "pk/cases/PKLHC",  # Lahore High Court
        "pk/cases/PKSHC",  # Sindh High Court
        "pk/cases/PKFSC",  # Federal Shariat Court
    ]

    def search_keyword(self, keyword: str, max_pages: int = 3) -> list[dict]:
        results = []
        log.info(f"[WorldLII] Searching: '{keyword}'")
        for db in self.DATABASES:
            for page in range(0, max_pages * 20, 20):
                resp = polite_get(self.SEARCH_URL, params={
                    "method": "boolean",
                    "query": keyword,
                    "meta": f"/{db}",
                    "results": 20,
                    "rank": "off",
                    "start": page,
                })
                if not resp:
                    break
                soup = BeautifulSoup(resp.text, "lxml")
                items = self._parse_results(soup)
                if not items:
                    break
                results.extend(items)
                log.info(f"[WorldLII] {db} page {page//20+1}: +{len(items)} cases")
        return results

    def fetch_full_text(self, url: str) -> str:
        if not url or url == "#":
            return ""
        resp = polite_get(url, delay=3)
        if not resp:
            return ""
        soup = BeautifulSoup(resp.text, "lxml")
        # WorldLII uses a specific content div
        for sel in ["div#content", "div.case", "div#judgment", "body"]:
            el = soup.select_one(sel)
            if el and len(el.get_text()) > 200:
                return el.get_text("\n", strip=True)[:5000]
        return ""

    def _parse_results(self, soup: BeautifulSoup) -> list[dict]:
        items = []
        # WorldLII returns results in <dt> and <dd> pairs
        links = soup.select("ol li a, dt a, .result a")
        for link in links:
            href = link.get("href", "")
            if not href or "sinosrch" in href:
                continue
            full_url = urljoin(self.BASE, href)
            title    = link.get_text(strip=True)
            if len(title) < 10:
                continue

            # Determine court from URL path
            court_code = "SHC" if "PKSHC" in href else \
                         "LHC" if "PKLHC" in href else \
                         "SC"  if "PKSC"  in href else "PKT"
            court_name = {
                "SHC": "Sindh High Court",
                "LHC": "Lahore High Court",
                "SC":  "Supreme Court of Pakistan",
                "PKT": "Pakistan Tribunal",
            }.get(court_code, court_code)

            # Extract year from URL e.g. /2019/12.html
            yr_m = re.search(r"/(\d{4})/", href)
            year = yr_m.group(1) if yr_m else ""

            items.append({
                "court": court_code, "court_name": court_name,
                "citation": parse_citation(title),
                "case_number": parse_case_number(title),
                "title": title[:250],
                "url": full_url,
                "date": year,
                "full_text": "",
                "scraped_at": datetime.now().isoformat(),
                "source": "WorldLII",
            })
        return items


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

SCRAPERS = {
    "SHC":      SHCScraper,
    "LHC":      LHCScraper,
    "IHC":      IHCScraper,
    "WorldLII": WorldLIIScraper,
}


def scrape_all_fields(courts: list[str] = None, max_pages_per_keyword: int = 3,
                      fetch_full_text: bool = True, quick: bool = False) -> list[dict]:
    """
    Scrape ALL legal fields across all courts.
    This is the main mass-scraping function — can collect thousands of cases.

    Args:
        courts:                  List of court codes to scrape (default: all)
        max_pages_per_keyword:   Pages to scrape per keyword per court
        fetch_full_text:         Whether to fetch full judgment text
        quick:                   If True, only fetch recent judgments (no keyword loop)
    """
    if courts is None:
        courts = list(SCRAPERS.keys())

    all_cases = []
    seen_urls = set()
    seen_titles = set()

    keywords = ALL_LEGAL_FIELDS if not quick else []

    for court_code in courts:
        if court_code not in SCRAPERS:
            log.warning(f"Unknown court: {court_code}")
            continue

        scraper = SCRAPERS[court_code]()
        court_cases = []

        # 1. Fetch recent judgments first
        log.info(f"\n{'='*60}")
        log.info(f"[{court_code}] Starting scrape...")
        try:
            recent = scraper.fetch_recent(max_pages=10 if not quick else 5)
            court_cases.extend(recent)
            log.info(f"[{court_code}] Recent: {len(recent)} cases")
        except AttributeError:
            pass  # WorldLII doesn't have fetch_recent
        except Exception as e:
            log.error(f"[{court_code}] Recent fetch error: {e}")

        # 2. Search all legal fields
        for keyword in keywords:
            try:
                kw_cases = scraper.search_keyword(keyword, max_pages=max_pages_per_keyword)
                court_cases.extend(kw_cases)
                log.info(f"[{court_code}] '{keyword[:40]}': {len(kw_cases)} cases")
            except Exception as e:
                log.error(f"[{court_code}] Keyword '{keyword}' error: {e}")

        # 3. Deduplicate
        unique = []
        for c in court_cases:
            key = c.get("url") or c.get("title", "")[:100]
            if key and key not in seen_urls and key not in seen_titles:
                seen_urls.add(key)
                seen_titles.add(c.get("title", "")[:100])
                unique.append(c)
        log.info(f"[{court_code}] Unique cases after dedup: {len(unique)}")

        # 4. Fetch full text
        if fetch_full_text and hasattr(scraper, "fetch_full_text"):
            for i, case in enumerate(unique):
                if case.get("url") and case.get("url") != "#":
                    log.info(f"[{court_code}] Full text {i+1}/{len(unique)}: {case['url'][:70]}")
                    case["full_text"] = scraper.fetch_full_text(case["url"])

        all_cases.extend(unique)
        log.info(f"[{court_code}] Done. Running total: {len(all_cases)} cases")

    # Save to disk
    if all_cases:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = DATA_DIR / f"cases_{ts}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(all_cases, f, ensure_ascii=False, indent=2)
        log.info(f"\n✅ Saved {len(all_cases)} cases → {out}")

    return all_cases


def scrape_keyword(keyword: str, courts: list[str] = None,
                   max_pages: int = 5, fetch_full_text: bool = True) -> list[dict]:
    """Scrape a single keyword across specified courts."""
    if courts is None:
        courts = list(SCRAPERS.keys())

    all_cases = []
    seen = set()

    for court_code in courts:
        if court_code not in SCRAPERS:
            continue
        scraper = SCRAPERS[court_code]()
        try:
            cases = scraper.search_keyword(keyword, max_pages=max_pages)
            if fetch_full_text and hasattr(scraper, "fetch_full_text"):
                for c in cases:
                    if c.get("url"):
                        c["full_text"] = scraper.fetch_full_text(c["url"])
            for c in cases:
                key = c.get("url") or c.get("title", "")[:80]
                if key not in seen:
                    seen.add(key)
                    all_cases.append(c)
        except Exception as e:
            log.error(f"[{court_code}] Error: {e}")

    if all_cases:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = DATA_DIR / f"cases_{ts}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(all_cases, f, ensure_ascii=False, indent=2)
        log.info(f"✅ Saved {len(all_cases)} cases → {out}")

    return all_cases


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pakistan High Court mass scraper")
    parser.add_argument("keyword", nargs="?", default=None,
                        help="Specific keyword to search (optional)")
    parser.add_argument("--courts", nargs="+", default=["SHC", "LHC", "IHC", "WorldLII"],
                        help="Courts to scrape: SHC LHC IHC WorldLII")
    parser.add_argument("--pages", type=int, default=3,
                        help="Pages per keyword per court (default: 3)")
    parser.add_argument("--no-fulltext", action="store_true",
                        help="Skip fetching full judgment text (faster)")
    parser.add_argument("--quick", action="store_true",
                        help="Only fetch recent judgments, skip keyword loop")
    args = parser.parse_args()

    fetch_ft = not args.no_fulltext

    if args.keyword:
        cases = scrape_keyword(args.keyword, courts=args.courts,
                               max_pages=args.pages, fetch_full_text=fetch_ft)
    else:
        cases = scrape_all_fields(courts=args.courts,
                                  max_pages_per_keyword=args.pages,
                                  fetch_full_text=fetch_ft,
                                  quick=args.quick)

    print(f"\n✅ Total cases scraped: {len(cases)}")
    print(f"   Data saved to: {DATA_DIR}")
    print(f"\nNext step — index into search DB:")
    print(f"   python embeddings/vector_db.py")
