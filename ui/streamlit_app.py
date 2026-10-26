import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd
import streamlit as st
from app import llm
from app.service import Copilot
from app.sql_agent import run_readonly

try:                                  # Streamlit Cloud secrets -> env vars (ANTHROPIC_API_KEY, LLM_DAILY_CAP, ...)
    for _k in st.secrets:
        if isinstance(st.secrets[_k], str):
            os.environ.setdefault(_k, st.secrets[_k])
except Exception:
    pass

st.set_page_config(page_title="Enterprise Intelligence Copilot", page_icon="🧭", layout="wide")
MAX_Q = 40   # per-session cap for the public demo

EXAMPLES = {
    "📊 Analytics (SQL)": ["Which category revenue dropped most last quarter?", "Revenue by state",
                          "Which payment method is used most?", "Do late deliveries hurt review scores?",
                          "Top sellers by revenue", "Show monthly revenue trend"],
    "💬 Customer reviews": ["What do customers complain about?", "What do customers complain about in health_beauty?",
                           "Which categories have the worst review scores?"],
    "📄 Documents (RAG)": ["How quickly must a grievance officer acknowledge a consumer complaint?",
                          "Can platforms charge fees when a buyer cancels an order?",
                          "How many employees does Amazon have?", "What does Amazon say about seasonality?",
                          "What is the capital of France?"],
}


@st.cache_resource(show_spinner="Indexing documents…")
def load():
    return Copilot()


@st.cache_data(show_spinner=False)
def q(sql):
    return run_readonly(sql)


cp = load()

with st.sidebar:
    st.title("🧭 Copilot")
    role = st.selectbox("Your role (access control demo)", ["analyst", "viewer", "admin"],
                        help="viewer: documents only. analyst/admin: documents + SQL analytics + review intelligence.")
    mode = st.selectbox("Retrieval mode", ["hybrid", "bm25", "dense"], help="hybrid = BM25 + dense fused with RRF")
    st.caption(f"{len(cp.chunks)} chunks indexed • embedder: `{cp.retriever.emb.name}` • "
               f"LLM: {'on' if llm.available() else 'off (extractive answers)'}")
    st.markdown("---")
    st.caption("**Data:** Olist Brazilian E-Commerce (Kaggle, CC BY-NC-SA 4.0) · Amazon 10-K FY2025 (SEC) · "
               "Consumer Protection (E-Commerce) Rules 2020 (India, via TaxGuru). Portfolio demo, not legal or financial advice.")

st.title("Enterprise Intelligence Copilot")
st.caption("Ask questions in plain English. Questions are routed to document search (RAG with citations), "
           "safe read-only SQL over 99k Olist orders, or customer-review analysis.")
t_ask, t_dash, t_eval, t_how = st.tabs(["💬 Ask", "📊 Dashboard", "🧪 Evaluation", "🏗️ How it works"])

# ------------------------------------------------------------------ ASK
with t_ask:
    st.session_state.setdefault("msgs", [])
    st.session_state.setdefault("n", 0)
    pending = None
    for group, qs in EXAMPLES.items():
        st.markdown(f"**{group}**")
        cols = st.columns(3)
        for i, ex in enumerate(qs):
            if cols[i % 3].button(ex, key=f"ex-{ex}", width="stretch"):
                pending = ex
    typed = st.chat_input("Ask about sales, reviews, Amazon's 10-K or e-commerce rules…")
    question = typed or pending
    if question:
        if st.session_state.n >= MAX_Q:
            st.warning("Demo limit reached for this session. Reload the page to continue.")
        else:
            st.session_state.n += 1
            st.session_state.msgs.append(("user", question, None))
            st.session_state.msgs.append(("assistant", None, cp.ask(question, role, mode)))
    for who, text, res in reversed([m for m in st.session_state.msgs]) if False else st.session_state.msgs[-6:]:
        if who == "user":
            st.chat_message("user").write(text)
            continue
        with st.chat_message("assistant"):
            st.write(res["answer"])
            if res.get("abstained") and not res.get("citations") and res["route"] == "docs":
                st.info("The system abstained instead of guessing: no sufficiently relevant evidence was found.")
            if res.get("citations"):
                st.markdown("**Sources**")
                for c in res["citations"]:
                    st.caption(f"📎 {c}")
            if res.get("table"):
                df = pd.DataFrame(res["table"])
                if "month" in df.columns:
                    st.line_chart(df.set_index("month")[[c for c in df.columns if c in ("revenue", "avg_score")][:1]])
                else:
                    num = [c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c])]
                    if num and len(df) > 1:
                        st.bar_chart(df.set_index(df.columns[0])[num[:1]])
                st.dataframe(df, width="stretch", hide_index=True)
            if res.get("details"):
                for th in res["details"]["themes"][:4]:
                    with st.expander(f"{th['theme']} — {th['reviews']:,} reviews ({th['share_pct']}%)"):
                        for e in th["evidence"]:
                            st.caption(f"⭐{e['score']} · {e['category']} · #{e['review_id']}")
                            st.write(f"“{e['text']}”")
            with st.expander("How this was answered"):
                for t in res["trace"]:
                    st.write("• " + t)
                if res.get("sql"):
                    st.code(res["sql"], language="sql")
                    st.caption(res.get("definition", "") + f" Data ends {res.get('data_end')}.")
                for p in res.get("passages", []):
                    st.caption(f"{p['label']} (score {p['score']})")
                    st.write(p["text"] + "…")
            st.caption(f"route: **{res['route']}** · {res['latency_ms']} ms · role: {res['role']}")

# ------------------------------------------------------------------ DASHBOARD
with t_dash:
    k = q("""SELECT ROUND(SUM(i.price)) AS revenue, COUNT(DISTINCT o.order_id) AS orders FROM order_items i
             JOIN orders o ON o.order_id = i.order_id WHERE o.status NOT IN ('canceled','unavailable')""").iloc[0]
    r = q("SELECT ROUND(AVG(score),2) AS s, ROUND(100.0*AVG(score<=2),1) AS neg FROM reviews").iloc[0]
    l = q("SELECT ROUND(100.0*AVG(is_late),1) AS late FROM orders WHERE status='delivered' AND is_late IS NOT NULL").iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("Revenue (BRL)", f"{k.revenue:,.0f}"); b.metric("Orders", f"{int(k.orders):,}")
    c.metric("Avg review score", f"{r.s}/5", f"{r.neg}% negative", delta_color="off")
    d.metric("Late deliveries", f"{l.late}%")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Monthly revenue")
        m = q("""SELECT o.purchase_month AS month, ROUND(SUM(i.price)) AS revenue FROM order_items i JOIN orders o ON o.order_id=i.order_id
                 WHERE o.status NOT IN ('canceled','unavailable') AND o.purchase_month BETWEEN '2017-01' AND '2018-08'
                 GROUP BY 1 ORDER BY 1""")
        st.line_chart(m.set_index("month"))
        st.subheader("Average review score by month")
        t = q("""SELECT substr(created_date,1,7) AS month, ROUND(AVG(score),2) AS avg_score FROM reviews
                 WHERE created_date BETWEEN '2017-01-01' AND '2018-08-31' GROUP BY 1 ORDER BY 1""")
        st.line_chart(t.set_index("month"))
    with c2:
        st.subheader("Top 10 categories by revenue")
        cat = q("""SELECT p.category, ROUND(SUM(i.price)) AS revenue FROM order_items i JOIN orders o ON o.order_id=i.order_id
                   JOIN products p ON p.product_id=i.product_id WHERE o.status NOT IN ('canceled','unavailable')
                   GROUP BY 1 ORDER BY 2 DESC LIMIT 10""")
        st.bar_chart(cat.set_index("category"))
        st.subheader("Late-delivery rate by state (300+ orders)")
        ls = q("""SELECT c.state, ROUND(100.0*AVG(o.is_late),1) AS late_pct FROM orders o JOIN customers c ON c.customer_id=o.customer_id
                  WHERE o.status='delivered' AND o.is_late IS NOT NULL GROUP BY 1 HAVING COUNT(*)>=300 ORDER BY 2 DESC LIMIT 10""")
        st.bar_chart(ls.set_index("state"))
    st.caption("Revenue = sum of item prices (excl. freight) for orders not canceled/unavailable. Olist data covers 2016–2018.")

# ------------------------------------------------------------------ EVAL
with t_eval:
    f = ROOT / "eval" / "results.json"
    if f.exists():
        R = json.loads(f.read_text())
        st.markdown(f"Measured on **{R['corpus']['questions']} labelled questions** + {R['corpus']['negatives']} out-of-scope "
                    f"questions · {R['corpus']['chunks']} chunks · embedder `{R['corpus']['embedder']}`. "
                    "Reproduce with `python eval/run_eval.py`.")
        st.subheader("Retrieval: baseline vs improvements")
        st.dataframe(pd.DataFrame(R["retrieval"]).T, width="stretch")
        a1, a2, a3 = st.columns(3)
        a1.metric("Text-to-SQL vs pandas truth", f"{R['sql']['passed']}/{R['sql']['total']}")
        a2.metric("Out-of-scope correctly abstained", f"{R['answers']['correct_abstain_rate_out_of_scope']:.0%}")
        a3.metric("SQL attacks blocked", R["security"]["sql_attacks_blocked"])
        st.info("**Honest reading:** hybrid ≈ BM25 here, and the lexical reranker did not help. The 'dense' retriever in "
                "this build is TF-IDF, not neural embeddings. Neural embeddings and a cross-encoder are the next experiment. "
                f"False-abstain rate on in-scope questions: {R['answers']['false_abstain_rate']:.0%}.")
        with st.expander("SQL checks (each compared with pandas computed from the raw CSVs)"):
            st.dataframe(pd.DataFrame(R["sql"]["checks"]).astype(str), width="stretch", hide_index=True)
    else:
        st.warning("Run `python eval/run_eval.py` to generate results.")

# ------------------------------------------------------------------ HOW
with t_how:
    st.graphviz_chart("""digraph G { rankdir=LR; node [shape=box, style=rounded];
      Q [label="Question + role"]; R [label="Router"];
      D [label="Document RAG\\nBM25 + dense → RRF\\n→ grounded answer"]; S [label="Text-to-SQL\\nvalidator → authorizer → read-only"];
      V [label="Review intelligence\\nPortuguese theme rules"];
      A [label="Answer + citations / SQL + trace\\nor abstain"];
      Q -> R; R -> D; R -> S; R -> V; D -> A; S -> A; V -> A; }""")
    st.markdown("""
**Safety:** read-only SQLite connection · single `SELECT` only · table allow-list · engine-level SQLite *authorizer* ·
row limit + timeout · role-based document filtering **before** ranking · prompt-injection filter on retrieved text ·
abstains when evidence is missing.

**Known limitations (deliberate honesty):**
- Answers are *extractive* unless an LLM key is configured; extractive answers can be incomplete.
- "Dense" retrieval is TF-IDF in this deployment (light enough for free hosting). Neural embeddings: `EMBEDDER=fastembed`.
- Text-to-SQL uses vetted templates (every one verified against pandas); free-form questions fall back to an LLM only if configured.
- Review complaint themes are keyword rules over Portuguese text, not a trained classifier.
- Olist data ends Aug 2018, so "last quarter" means **2018-Q2**.
""")
