"""Orchestrator: route -> engine -> trace. Logs route/latency to JSONL (best effort)."""
import json, math, re, time
from pathlib import Path
from .ingest import ingest_dir
from .embeddings import get_embedder
from .retrieval import HybridRetriever, tokenize
from .rerank import get_reranker, LexicalReranker
from .answer import build_answer, is_injection, ABSTAIN
from .router import route
from .sql_agent import answer_sql, UnsafeSQL
from .reviews import analyze_reviews, summarize
from .entities import parse
from .sql_agent import categories
from . import llm

ROOT = Path(__file__).resolve().parent.parent
COVERAGE_MIN = 0.40      # idf-weighted share of question terms the best passage must contain, else abstain (tuned on eval set: optimistic)
RANKING = re.compile(r"(worst|best|lowest|highest).{0,40}(review|rating|score)", re.I)


class Copilot:
    def __init__(self, docs_dir=None, chunk_size=120, overlap=30):
        self.docs_dir = Path(docs_dir) if docs_dir else ROOT / "data" / "docs"
        self.chunks = ingest_dir(self.docs_dir, chunk_size, overlap)
        self.retriever = HybridRetriever(self.chunks, get_embedder())
        self.reranker = get_reranker()
        self._lex = LexicalReranker()

    def search_docs(self, query, role="viewer", rerank=True, k=5, mode="hybrid"):
        hits = self.retriever.search(query, role=role, k=max(k, 8) if rerank else k, mode=mode)
        return self._lex.rerank(query, hits, k=k) if rerank else hits

    def _coverage(self, query, text):
        bm, qt = self.retriever.bm25, set(tokenize(query))
        if not qt:
            return 0.0
        top = math.log(1 + (len(bm.toks) + 0.5) / 0.5)           # unseen words get the maximum idf
        ct = set(tokenize(text))
        tot = sum(bm.idf.get(w, top) for w in qt)
        return sum(bm.idf.get(w, top) for w in qt if w in ct) / tot

    def ask(self, question, role="viewer", mode="hybrid"):
        t0, (r, scores) = time.time(), route(question)
        trace = [f"Router → **{r}** (scores: {scores})", f"Role: {role}"]
        try:
            if r in ("sql", "reviews") and role == "viewer":
                res = {"answer": "Access denied: analytics and review intelligence require the analyst or admin role.", "abstained": True}
                trace.append("Blocked by role-based access control")
            elif r == "sql" or (r == "reviews" and RANKING.search(question)):
                res = answer_sql(question)
                trace += [f"Template/intent: {res.get('intent', res.get('sql_source', '-'))}", "SQL validated → authorizer → read-only run"]
            elif r == "reviews":
                cat = parse(question, categories())["category"]
                a = analyze_reviews(cat)
                res = {"answer": summarize(a), "details": a, "abstained": False}
                trace.append(f"Complaint themes over negative reviews (scope: {a['scope']})")
            else:
                hits = self.search_docs(question, role, rerank=self.reranker is not None, k=5, mode=mode)
                res = {**ABSTAIN}
                cov = max((self._coverage(question, h["chunk"].text) for h in hits[:3]), default=0.0)
                trace.append(f"Retrieved {len(hits)} passages ({mode}); best term coverage {cov:.2f} (min {COVERAGE_MIN})")
                if hits and cov >= COVERAGE_MIN:
                    res = build_answer(question, hits[:3])
                res["passages"] = [{"label": h["chunk"].label, "score": round(h["score"], 4), "text": h["chunk"].text[:350]} for h in hits[:3]]
                if res["abstained"]:
                    trace.append("Abstained: insufficient evidence")
        except UnsafeSQL as e:
            res = {"answer": f"Blocked unsafe or invalid SQL: {e}", "abstained": True}
        except Exception as e:
            res = {"answer": f"Something went wrong ({type(e).__name__}). Please rephrase.", "abstained": True}
        res.update(route=r, role=role, trace=trace, latency_ms=round((time.time() - t0) * 1000, 1), input_flagged=is_injection(question))
        try:
            with open(ROOT / "data" / "query_log.jsonl", "a") as f:
                f.write(json.dumps({"q": question, "route": r, "role": role, "ms": res["latency_ms"], "abstained": res.get("abstained")}) + "\n")
        except OSError:
            pass
        return res
