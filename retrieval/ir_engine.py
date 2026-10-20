"""
Core IR engine — indexing plus the three retrieval models required
by the syllabus:
  1. Boolean retrieval      -> boolean_search()
  2. Vector Space retrieval -> vector_search()   (TF-IDF + cosine similarity)
  3. Probabilistic retrieval-> probabilistic_search()  (BM25)
"""
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from retrieval.query_refinement import expand_query

from core.models import Post


def _tokenize(text):
    return re.findall(r'\b[a-z]{2,}\b', text.lower())


class IREngine:
    """Loads all posts once, builds every index needed by the three
    retrieval models. Instantiate this once per request/search."""

    def __init__(self):
        self.posts = list(Post.objects.all())
        self.doc_texts = [f"{p.title} {p.content}" for p in self.posts]
        self.doc_ids = [p.id for p in self.posts]

        # --- Vector Space Model (TF-IDF) ---
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts) if self.doc_texts else None

        # --- Probabilistic Model (BM25) ---
        tokenized_docs = [_tokenize(t) for t in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized_docs) if tokenized_docs else None

        # --- Boolean Model (inverted index) ---
        self.inverted_index = {}
        for idx, tokens in enumerate(tokenized_docs):
            for term in set(tokens):
                self.inverted_index.setdefault(term, set()).add(idx)

    def boolean_search(self, query):
        tokens = query.strip().split()
        if not tokens:
            return []

        result_set = None
        operator = 'AND'
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok.upper() in ('AND', 'OR', 'NOT'):
                operator = tok.upper()
                i += 1
                continue
            term_docs = self.inverted_index.get(tok.lower(), set())
            if result_set is None:
                result_set = term_docs
            elif operator == 'AND':
                result_set = result_set & term_docs
            elif operator == 'OR':
                result_set = result_set | term_docs
            elif operator == 'NOT':
                result_set = result_set - term_docs
            i += 1

        result_set = result_set or set()
        return [self.posts[idx] for idx in result_set]

    def vector_search(self, query, top_k=10):
        if self.tfidf_matrix is None:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        ranked_idx = scores.argsort()[::-1][:top_k]
        return [(self.posts[i], float(scores[i])) for i in ranked_idx if scores[i] > 0]

    def probabilistic_search(self, query, top_k=10):
        if self.bm25 is None:
            return []
        query_tokens = _tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked_idx = scores.argsort()[::-1][:top_k]
        return [(self.posts[i], float(scores[i])) for i in ranked_idx if scores[i] > 0]
    def vector_search_refined(self, query, top_k=10):
        return self.vector_search(expand_query(query), top_k=top_k)

    def probabilistic_search_refined(self, query, top_k=10):
        return self.probabilistic_search(expand_query(query), top_k=top_k)