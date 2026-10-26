"""Embedders (all expose .name). Default TF-IDF = offline, zero download, but it is LEXICAL, not
semantic. EMBEDDER=fastembed uses a small ONNX model (BAAI/bge-small-en-v1.5, low RAM);
EMBEDDER=st uses sentence-transformers. Any failure falls back to TF-IDF (name tells the truth)."""
import os
import numpy as np


class TfidfEmbedder:
    name = "tfidf"

    def fit(self, texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.v = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True,
                                 max_features=30000, min_df=1)
        self.v.fit(texts)
        return self

    def encode(self, texts):
        return self.v.transform(texts)   # sparse, L2-normalised rows


class FastEmbedEmbedder:
    name = "fastembed bge-small-en-v1.5"

    def __init__(self):
        from fastembed import TextEmbedding
        self.m = TextEmbedding("BAAI/bge-small-en-v1.5")

    def fit(self, texts):
        return self

    def encode(self, texts):
        v = np.asarray(list(self.m.embed(list(texts))), dtype="float32")
        return v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-9)


class STEmbedder:
    name = "sentence-transformers MiniLM"

    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model)

    def fit(self, texts):
        return self

    def encode(self, texts):
        return np.asarray(self.m.encode(list(texts), normalize_embeddings=True), dtype="float32")


def get_embedder():
    kind = os.getenv("EMBEDDER", "tfidf").lower()
    try:
        if kind == "fastembed":
            return FastEmbedEmbedder()
        if kind == "st":
            return STEmbedder()
    except Exception as e:  # noqa
        print(f"[embeddings] {kind} unavailable ({e}); falling back to TF-IDF")
    return TfidfEmbedder()
