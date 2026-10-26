# 🧭 Enterprise Intelligence Copilot

**Live demo:** _add your Streamlit URL here after deploying (see DEPLOY.md)_

Ask business questions in plain English. A router sends each question to one of three engines:

| Question type | Engine | Example |
|---|---|---|
| Documents | Hybrid RAG (BM25 + dense → RRF), cited, **abstains** without evidence | "How quickly must a grievance officer acknowledge a complaint?" |
| Analytics | Validated **read-only Text-to-SQL** over 99k real Olist orders | "Which category revenue dropped most last quarter?" |
| Customer voice | Complaint-theme analysis of 98k Portuguese reviews, with evidence quotes | "What do customers complain about in health_beauty?" |

**Data (all real):** [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (orders, items, products, customers, sellers, payments, reviews) ·
Amazon Form 10-K FY2025 (SEC, 144 pages) · Consumer Protection (E-Commerce) Rules 2020 (India).

## Results (measured, reproducible: `python eval/run_eval.py`)
37 hand-labelled questions over 554 chunks. A "hit" means a top-k chunk from the right document contains the answer text.

| retrieval method | hit@1 | hit@3 | hit@5 | MRR | p50 ms |
|---|---|---|---|---|---|
| bm25 | 0.73 | 0.784 | 0.838 | 0.776 | 1.6 |
| dense (tf-idf) | 0.514 | 0.73 | 0.757 | 0.619 | 1.6 |
| hybrid (RRF) | 0.676 | 0.757 | 0.838 | 0.735 | 1.7 |
| hybrid + lexical rerank | 0.622 | 0.757 | 0.784 | 0.706 | 2.7 |

- **Text-to-SQL:** 10/10 checks equal pandas ground truth computed independently from the raw CSVs.
- **Abstention:** 90% of 20 out-of-scope questions correctly refused; 16% of in-scope questions wrongly refused (threshold tuned on this same set, so treat as optimistic).
- **Security:** 25/25 SQL attack strings blocked; 5/5 prompt-injection patterns detected.

**What the numbers say (honestly):** BM25 alone is slightly *better* than hybrid at hit@1 here, and my lexical reranker *hurt*, so reranking is off by default. TF-IDF 'dense' is the weakest, as expected (it is lexical).
The "dense" retriever in this build is TF-IDF (light enough for free hosting); neural embeddings and a cross-encoder are the next experiment
(`EMBEDDER=fastembed` / `EMBEDDER=st RERANKER=ce`). Answers are extractive unless an LLM key is configured; LLM faithfulness is **not** measured yet.
Known failure, kept as an `xfail` test: "How many *people* does Amazon *employ*?" misses because the filing says "employees" (vocabulary gap).

## Architecture
```
PDF / DB ─► ingest (crop sidebars, drop headers/footers, section-aware chunking, dedupe, role metadata)
Question ─► router ─┬─► docs: BM25 + dense ─► RRF ─► (optional rerank) ─► grounded answer + citations | abstain
                    ├─► sql : template (or LLM fallback) ─► validator ─► SQLite authorizer ─► read-only exec ─► explanation
                    └─► reviews: Portuguese theme rules + evidence quotes
every call: role-based access control · latency/route logging · trace shown in the UI
```
SQL safety layers: single `SELECT` only · table allow-list · engine-level SQLite authorizer · read-only connection · row cap · timeout.
Document ACL is applied **before** ranking. Retrieved text is scanned for prompt-injection phrases.

## Run locally
```bash
pip install -r requirements-dev.txt
pytest -q                          # 36 pass, 1 documented xfail
python eval/run_eval.py            # regenerates eval/REPORT.md + results.json
streamlit run ui/streamlit_app.py  # UI
uvicorn app.api:app --reload       # optional REST API (/docs)
```
Rebuild the DB from raw Kaggle CSVs: `python scripts/build_olist_db.py path/to/csvs` (ground truth: `scripts/make_sql_truth.py`).

## Layout
`app/` ingest · embeddings · retrieval · rerank · answer · sql_agent · reviews · router · service · llm · api &nbsp;|&nbsp; `ui/` Streamlit &nbsp;|&nbsp; `eval/` questions, ground truth, runner, report &nbsp;|&nbsp; `tests/` pytest (+ synthetic ACL/injection fixtures)

## Limitations
37 eval questions · extractive answers by default · review themes are keyword rules, not a trained model · revenue excludes freight ·
Olist ends Aug 2018 so "last quarter" = 2018-Q2 · demo is not legal/financial advice · Olist is CC BY-NC-SA 4.0 (non-commercial, attribution).

## Resume bullets (every claim below is measured in this repo)
- Built a RAG + Text-to-SQL analytics copilot over 99k real e-commerce orders and two real documents (144-page SEC 10-K, India e-commerce rules); deployed publicly with Streamlit.
- Verified SQL answers against independent pandas ground truth (10/10); layered SQL safety (validator + engine authorizer + read-only) blocked 25/25 attack queries.
- Benchmarked BM25, TF-IDF-dense, RRF hybrid and reranking on 37 labelled questions; found the reranker hurt hit@1 and removed it.
