import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from database import get_connection

log = logging.getLogger(__name__)


class CaseLawVectorDB:

    def __init__(self):
        self.cases = []
        self.vectorizer = TfidfVectorizer(max_df=0.9, min_df=1)
        self._matrix = None

    def load_from_db(self):
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("SELECT court, title, citation, date, url, full_text FROM cases")
        rows = cur.fetchall()

        self.cases = [
            {
                "court": r[0],
                "title": r[1],
                "citation": r[2],
                "date": r[3],
                "url": r[4],
                "full_text": r[5],
            }
            for r in rows
        ]

        if self.cases:
            docs = [self._doc(c) for c in self.cases]
            self._matrix = self.vectorizer.fit_transform(docs)

        conn.close()

    def _doc(self, c):
        return " ".join([
            c.get("title", ""),
            c.get("citation", ""),
            c.get("date", "")
        ])

    def search(self, query, top_k=10):
        if self._matrix is None:
            return []

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix)[0]

        idxs = np.argsort(sims)[::-1]

        return [self.cases[i] for i in idxs[:top_k]]

    def count(self):
        return len(self.cases)


_db = None

def get_db():
    global _db
    if not _db:
        _db = CaseLawVectorDB()
    return _db
