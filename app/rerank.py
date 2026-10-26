"""Reranking. Default = lexical-overlap reranker (offline). RERANKER=ce uses a cross-encoder."""
import os
from .retrieval import tokenize


class LexicalReranker:
    def rerank(self, query, hits, k=3):
        qt = tokenize(query)
        q = set(qt)
        qb = set(zip(qt, qt[1:]))
        def score(h):
            tt = tokenize(h["chunk"].text)
            t = set(tt)
            bi = len(qb & set(zip(tt, tt[1:]))) / (len(qb) or 1)   # phrase match bonus
            return len(q & t) / (len(q) or 1) + 0.5 * bi + 0.1 * h["score"]
        out = sorted(hits, key=score, reverse=True)[:k]
        for h in out:
            h["rerank"] = score(h)
        return out


class CrossEncoderReranker:
    def __init__(self, model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.m = CrossEncoder(model)

    def rerank(self, query, hits, k=3):
        if not hits:
            return []
        s = self.m.predict([(query, h["chunk"].text) for h in hits])
        for h, v in zip(hits, s):
            h["rerank"] = float(v)
        return sorted(hits, key=lambda h: -h["rerank"])[:k]


def get_reranker():
    """None (default) = no reranking: on this corpus the lexical reranker LOWERED hit@1 (see eval/REPORT.md).
    RERANKER=ce -> cross-encoder (needs sentence-transformers); RERANKER=lexical -> LexicalReranker."""
    r = os.getenv("RERANKER", "")
    return CrossEncoderReranker() if r == "ce" else LexicalReranker() if r == "lexical" else None
