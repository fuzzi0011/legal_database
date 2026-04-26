import pickle
import logging
from pathlib import Path
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DB_FILE = BASE_DIR / "embeddings" / "case_db.pkl"
DB_FILE.parent.mkdir(parents=True, exist_ok=True)


class CaseLawVectorDB:

    def __init__(self):
        self.cases = []
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_df=0.9,
            min_df=2
        )
        self._matrix = None
        self._load()

    # ─────────────────────────────

    def _save(self):
        with open(DB_FILE, "wb") as f:
            pickle.dump((self.cases, self.vectorizer, self._matrix), f)

    def _load(self):
        if DB_FILE.exists():
            try:
                with open(DB_FILE, "rb") as f:
                    self.cases, self.vectorizer, self._matrix = pickle.load(f)
                log.info(f"Loaded {len(self.cases)} cases")
            except:
                pass

    # ─────────────────────────────

    def _case_to_doc(self, c):
        return " ".join([
            c.get("title", ""),
            c.get("citation", ""),
            c.get("date", ""),
            c.get("full_text", "")[:3000]
        ])

    # ─────────────────────────────

    def _ingest_cases(self, new_cases):
        existing_urls = {c.get("url") for c in self.cases}

        filtered = []
        for c in new_cases:
            if c.get("url") and c["url"] not in existing_urls:
                filtered.append(c)

        if not filtered:
            return

        self.cases.extend(filtered)

        docs = [self._case_to_doc(c) for c in self.cases]
        self._matrix = self.vectorizer.fit_transform(docs)

        self._save()

        log.info(f"Added {len(filtered)} cases. Total: {len(self.cases)}")

    # ─────────────────────────────

    def search(self, query, top_k=10, court_filter=None):
        if self._matrix is None:
            return []

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix)[0]

        idxs = np.argsort(sims)[::-1]

        results = []

        for i in idxs:
            case = self.cases[i]

            if court_filter and case.get("court") != court_filter:
                continue

            results.append({
                "title": case.get("title"),
                "court": case.get("court"),
                "citation": case.get("citation"),
                "date": case.get("date"),
                "url": case.get("url"),
                "snippet": case.get("full_text", "")[:500],
                "similarity": float(sims[i])
            })

            if len(results) >= top_k:
                break

        return results

    # ─────────────────────────────

    def count(self):
        return len(self.cases)


_db_instance = None

def get_db():
    global _db_instance
    if not _db_instance:
        _db_instance = CaseLawVectorDB()
    return _db_instance
