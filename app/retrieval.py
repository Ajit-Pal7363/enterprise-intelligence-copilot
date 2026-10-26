"""Hybrid retrieval: BM25 + dense -> Reciprocal Rank Fusion, with role filtering."""
import math, re
from functools import lru_cache
from collections import Counter
import numpy as np
from nltk.stem import PorterStemmer
from scipy import sparse

TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("""the a an of to is are was were be been being am and or in for on at by as from with about into over
what which who whom whose when where why how do does did done have has had having can could should would will shall may might
many much any all each every some such it its this that these those there their they them i my me we our you your he she his her
not no than then also if so just very""".split())


_porter = PorterStemmer()


@lru_cache(maxsize=50000)
def stem(w):
    """Porter stem, then drop a trailing 'e' so employee/employees/employed all meet at 'employ'."""
    st = _porter.stem(w)
    return st[:-1] if st.endswith("e") and len(st) > 4 else st


def tokenize(t):
    return [stem(w) for w in TOKEN.findall(t.lower()) if w not in STOP]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.toks = [tokenize(d) for d in docs]
        self.tf = [Counter(t) for t in self.toks]
        self.len = np.array([len(t) for t in self.toks])
        self.avg = float(self.len.mean()) if len(self.len) else 1.0
        df = Counter(w for t in self.toks for w in set(t))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query):
        q = tokenize(query)
        s = np.zeros(len(self.toks))
        for i, tf in enumerate(self.tf):
            for w in q:
                if w in tf:
                    f = tf[w]
                    s[i] += self.idf[w] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


def rrf(rankings, k=60):
    """rankings: list of lists of doc indices (best first). Returns {idx: fused_score}."""
    out = {}
    for r in rankings:
        for rank, idx in enumerate(r):
            out[idx] = out.get(idx, 0) + 1 / (k + rank + 1)
    return out


class HybridRetriever:
    def __init__(self, chunks, embedder):
        self.chunks = chunks
        texts = [c.index_text for c in chunks]
        self.bm25 = BM25(texts)
        self.emb = embedder.fit(texts)
        self.matrix = self.emb.encode(texts)

    def _dense(self, query):
        q = self.emb.encode([query])
        s = self.matrix @ q.T if sparse.issparse(self.matrix) else self.matrix @ q[0]
        return np.asarray(s.todense() if sparse.issparse(s) else s).ravel()

    def search(self, query, role="viewer", k=5, mode="hybrid", pool=20):
        allowed = [i for i, c in enumerate(self.chunks) if role in c.allowed_roles]  # ACL BEFORE ranking
        if not allowed:
            return []
        bm = self.bm25.scores(query)
        dn = self._dense(query)
        rank_bm = [i for i in sorted(allowed, key=lambda i: -bm[i]) if bm[i] > 0][:pool]
        rank_dn = [i for i in sorted(allowed, key=lambda i: -dn[i]) if dn[i] > 0][:pool]
        if mode == "bm25":
            fused = {i: bm[i] for i in rank_bm}
        elif mode == "dense":
            fused = {i: dn[i] for i in rank_dn}
        else:
            fused = rrf([rank_bm, rank_dn])
        top = sorted(fused, key=lambda i: -fused[i])[:k]
        return [{"chunk": self.chunks[i], "score": float(fused[i]),
                 "bm25": float(bm[i]), "dense": float(dn[i])} for i in top]
