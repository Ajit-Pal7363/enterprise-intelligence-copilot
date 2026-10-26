"""Safe Text-to-SQL over the Olist database.
Defence in depth: (1) regex validator, (2) read-only SQLite connection, (3) SQLite *authorizer* (SELECT on allow-listed
tables only), (4) timeout, (5) row limit. Generation: audited templates (each verified vs pandas); Claude only if configured."""
import functools, json, re, sqlite3, time
import pandas as pd
from . import llm
from .db import connect_ro, get_db_path
from .entities import parse

SCHEMA = """orders(order_id, customer_id, customer_state, status, purchase_date, purchase_month, delivered_date, estimated_date, delivery_days, is_late)
customers(customer_id, state)
order_items(order_id, product_id, seller_id, price, freight_value)
products(product_id, category)   -- English category, e.g. bed_bath_table
payments(order_id, payment_type, payment_installments, payment_value)
reviews(review_id, order_id, score, comment, created_date)   -- score 1..5, comment is Portuguese
sellers(seller_id, seller_state, seller_city)"""
ALLOWED_TABLES = {"orders", "customers", "order_items", "products", "payments", "reviews", "sellers"}
FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|truncate|vacuum|exec|grant|load_extension)\b|sqlite_", re.I)
DEFINITION = "Revenue = sum of item prices (excluding freight) for orders not canceled/unavailable."


class UnsafeSQL(ValueError):
    pass


def validate_sql(sql: str, max_len=3000) -> str:
    s = (sql or "").strip().rstrip(";").strip()
    if not s or len(s) > max_len:
        raise UnsafeSQL("empty or too long")
    if ";" in s:
        raise UnsafeSQL("multiple statements not allowed")
    if "--" in s or "/*" in s:
        raise UnsafeSQL("comments not allowed")
    if not re.match(r"^(select|with)\b", s, re.I):
        raise UnsafeSQL("only SELECT queries allowed")
    if FORBIDDEN.search(s):
        raise UnsafeSQL("forbidden keyword")
    ctes = {c.lower() for c in re.findall(r"\b(\w+)\s+as\s*\(", s, re.I)}
    tables = {t.lower() for t in re.findall(r"\b(?:from|join)\s+([A-Za-z_]\w*)", s, re.I)}
    bad = tables - ALLOWED_TABLES - ctes
    if bad:
        raise UnsafeSQL(f"table not allowed: {sorted(bad)}")
    return s


def _authorizer(action, a1, a2, db, src):
    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION):
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        return sqlite3.SQLITE_OK if a1 in ALLOWED_TABLES else sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_DENY


def run_readonly(sql, params=(), max_rows=1000, timeout_s=5.0):
    sql = validate_sql(sql)
    con = connect_ro()
    con.set_authorizer(_authorizer)
    start = time.time()
    con.set_progress_handler(lambda: 1 if time.time() - start > timeout_s else 0, 20000)
    try:
        cur = con.execute(sql, params)
        rows = cur.fetchmany(max_rows)
        return pd.DataFrame(rows, columns=[d[0] for d in cur.description])
    except sqlite3.DatabaseError as e:      # authorizer denial / interrupted / bad column etc.
        raise UnsafeSQL(f"query rejected by the database engine: {e}")
    finally:
        con.close()


@functools.lru_cache(maxsize=1)
def categories():
    with connect_ro() as c:
        return tuple(r[0] for r in c.execute("SELECT DISTINCT category FROM products"))


def _qbounds(label):
    y, q = label.split("-Q")
    m = 3 * (int(q) - 1) + 1
    nxt = (int(y) + (m + 3 > 12), (m + 3 - 1) % 12 + 1)
    return f"{y}-{m:02d}-01", f"{nxt[0]}-{nxt[1]:02d}-01"


def periods(con):
    """Last two COMPLETE quarters. A month counts as complete if it has >=1000 orders (Olist is sparse at both ends)."""
    full = sorted(m for m, n in con.execute("SELECT purchase_month, COUNT(*) FROM orders GROUP BY 1") if m and n >= 1000)
    qs = {}
    for m in full:
        qs.setdefault(f"{m[:4]}-Q{(int(m[5:]) - 1) // 3 + 1}", []).append(m)
    done = [q for q in sorted(qs) if len(qs[q]) == 3]
    last, prev = done[-1], done[-2]
    (ps, pe), (ls, le) = _qbounds(prev), _qbounds(last)
    return {"last_label": last, "prev_label": prev, "prev_start": ps, "prev_end": pe, "last_start": ls, "last_end": le,
            "data_end": full[-1]}


@functools.lru_cache(maxsize=1)
def _periods():
    with connect_ro() as c:
        return periods(c)


BASE = ("FROM order_items oi JOIN orders o ON o.order_id = oi.order_id JOIN products p ON p.product_id = oi.product_id "
        "WHERE o.status NOT IN ('canceled','unavailable')")


def _filters(e, use=("year", "state", "category")):
    sql, params, applied = "", [], {}
    if "year" in use and e["year"]:
        sql += " AND substr(o.purchase_date,1,4) = ?"; params.append(e["year"]); applied["year"] = e["year"]
    if "state" in use and e["state"]:
        sql += " AND o.customer_state = ?"; params.append(e["state"]); applied["state"] = e["state"]
    if "category" in use and e["category"]:
        sql += " AND p.category = ?"; params.append(e["category"]); applied["category"] = e["category"]
    return sql, params, applied


def _rank(e, default_n=10, default_desc=True):
    return ("DESC" if default_desc and not e["ascending"] else "ASC"), (e["topn"] or default_n)


def build_template(q: str):
    e = parse(q, categories())
    has = lambda p: re.search(p, q.lower())
    deliv, score = has(r"deliver|late\b|delay|on[- ]time"), has(r"review|score|rating|satisf|stars?\b")
    if deliv and score:
        sql = ("SELECT CASE WHEN o.is_late = 1 THEN 'late' ELSE 'on time' END AS delivery, COUNT(*) AS reviews, "
               "ROUND(AVG(r.score),2) AS avg_score FROM reviews r JOIN orders o ON o.order_id = r.order_id "
               "WHERE o.status = 'delivered' AND o.is_late IS NOT NULL GROUP BY 1 ORDER BY 1 DESC")
        return dict(intent="late_vs_score", title="Review score: late vs on-time deliveries", sql=sql, params=[], applied={})
    if has(r"quarter|qoq|\b(drop|dropped|fell|fall|declin\w*|decreas\w*|gira|girawat)\b") and not deliv:
        p = _periods()
        grow = has(r"grew|growth|increase|rise|gain")
        sql = ("SELECT p.category AS category, "
               "ROUND(COALESCE(SUM(CASE WHEN o.purchase_date >= ? AND o.purchase_date < ? THEN oi.price END),0),2) AS prev_revenue, "
               "ROUND(COALESCE(SUM(CASE WHEN o.purchase_date >= ? AND o.purchase_date < ? THEN oi.price END),0),2) AS last_revenue, "
               "ROUND(COALESCE(SUM(CASE WHEN o.purchase_date >= ? AND o.purchase_date < ? THEN oi.price END),0) - "
               "COALESCE(SUM(CASE WHEN o.purchase_date >= ? AND o.purchase_date < ? THEN oi.price END),0),2) AS change "
               f"{BASE} AND p.category <> 'unknown' GROUP BY p.category HAVING prev_revenue >= 5000 "
               f"ORDER BY change {'DESC' if grow else 'ASC'} LIMIT {e['topn'] or 10}")
        params = [p["prev_start"], p["prev_end"], p["last_start"], p["last_end"],    # prev_revenue, last_revenue
                  p["last_start"], p["last_end"], p["prev_start"], p["prev_end"]]    # change = last - prev
        return dict(intent="quarterly_change", title=f"Category revenue change {p['prev_label']} → {p['last_label']}",
                    sql=sql, params=params, applied={"quarters": f"{p['prev_label']} vs {p['last_label']}"})
    if deliv:
        d, n = _rank(e, 10)
        f, pr, a = "", [], {}
        if e["year"]:
            f, pr, a = " AND substr(purchase_date,1,4) = ?", [e["year"]], {"year": e["year"]}
        sql = ("SELECT customer_state AS state, COUNT(*) AS delivered_orders, ROUND(AVG(delivery_days),1) AS avg_delivery_days, "
               "ROUND(100.0*AVG(is_late),1) AS late_pct FROM orders WHERE status='delivered' AND delivery_days IS NOT NULL "
               f"{f} GROUP BY customer_state HAVING COUNT(*) >= 200 ORDER BY late_pct {d} LIMIT {n}")
        return dict(intent="delivery", title="Delivery performance by state (≥200 delivered orders)", sql=sql, params=pr, applied=a)
    if has(r"review score|rating|satisf|stars?\b|average score"):
        d, n = _rank(e, 10, default_desc=False)
        if has(r"best|highest|top"):
            d = "DESC"
        sql = ("SELECT category, COUNT(*) AS reviews, ROUND(AVG(score),2) AS avg_score FROM (SELECT DISTINCT r.review_id, r.score, p.category "
               "FROM reviews r JOIN order_items oi ON oi.order_id = r.order_id JOIN products p ON p.product_id = oi.product_id "
               f"WHERE p.category <> 'unknown') GROUP BY category HAVING COUNT(*) >= 200 ORDER BY avg_score {d} LIMIT {n}")
        return dict(intent="review_score", title="Average review score by category (≥200 reviews)", sql=sql, params=[], applied={})
    if has(r"payment|install|credit|boleto|voucher|debit"):
        sql = ("SELECT payment_type, COUNT(DISTINCT order_id) AS orders, ROUND(SUM(payment_value),2) AS total_value, "
               "ROUND(100.0*SUM(payment_value)/(SELECT SUM(payment_value) FROM payments),1) AS pct_of_value, "
               "ROUND(AVG(payment_installments),1) AS avg_installments FROM payments GROUP BY payment_type ORDER BY total_value DESC")
        return dict(intent="payments", title="Payment methods", sql=sql, params=[], applied={})
    if has(r"status|cancel"):
        sql = ("SELECT status, COUNT(*) AS orders, ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM orders),2) AS pct "
               "FROM orders GROUP BY status ORDER BY orders DESC")
        return dict(intent="status", title="Orders by status", sql=sql, params=[], applied={})
    f, pr, a = _filters(e)
    d, n = _rank(e, 10)
    if has(r"\bsellers?\b"):
        sql = ("SELECT oi.seller_id, s.seller_state, ROUND(SUM(oi.price),2) AS revenue, COUNT(DISTINCT oi.order_id) AS orders "
               f"FROM order_items oi JOIN orders o ON o.order_id = oi.order_id JOIN products p ON p.product_id = oi.product_id "
               f"JOIN sellers s ON s.seller_id = oi.seller_id WHERE o.status NOT IN ('canceled','unavailable'){f} "
               f"GROUP BY oi.seller_id ORDER BY revenue {d} LIMIT {n}")
        return dict(intent="sellers", title="Sellers by revenue", sql=sql, params=pr, applied=a)
    if has(r"month|trend|over time|timeline|growth"):
        sql = ("SELECT o.purchase_month AS month, ROUND(SUM(oi.price),2) AS revenue, COUNT(DISTINCT oi.order_id) AS orders "
               f"{BASE} AND o.purchase_month BETWEEN '2017-01' AND '2018-08'{f} GROUP BY 1 ORDER BY 1")
        return dict(intent="monthly", title="Monthly revenue (Jan 2017 – Aug 2018)", sql=sql, params=pr, applied=a)
    if has(r"\bstates?\b|region|city|where"):
        f2, p2, a2 = _filters(e, ("year", "category"))
        sql = ("SELECT o.customer_state AS state, ROUND(SUM(oi.price),2) AS revenue, COUNT(DISTINCT oi.order_id) AS orders "
               f"{BASE}{f2} GROUP BY 1 ORDER BY revenue {d} LIMIT {n}")
        return dict(intent="states", title="Revenue by customer state", sql=sql, params=p2, applied=a2)
    if has(r"categor|product|segment"):
        sql = ("SELECT p.category, ROUND(SUM(oi.price),2) AS revenue, COUNT(DISTINCT oi.order_id) AS orders "
               f"{BASE}{f} GROUP BY 1 ORDER BY revenue {d} LIMIT {n}")
        return dict(intent="categories", title="Revenue by category", sql=sql, params=pr, applied=a)
    if has(r"revenue|sales|gmv|orders?\b|aov|average order|total"):
        sql = ("SELECT ROUND(SUM(oi.price),2) AS revenue, COUNT(DISTINCT oi.order_id) AS orders, "
               f"ROUND(SUM(oi.price)/COUNT(DISTINCT oi.order_id),2) AS avg_order_value {BASE}{f}")
        return dict(intent="overview", title="Revenue overview", sql=sql, params=pr, applied=a)
    return None


LLM_SYS = ("You write ONE read-only SQLite SELECT query for the schema below. Output JSON only: {\"sql\": \"...\"}. "
           "Rules: only SELECT/WITH; only these tables; no comments; no semicolons; LIMIT <= 200 unless aggregating; "
           "revenue = SUM(order_items.price) for orders with status NOT IN ('canceled','unavailable'). "
           "If the question cannot be answered from the schema output {\"sql\": null}.\n\nSCHEMA:\n" + SCHEMA)


def llm_sql(question):
    if not llm.available():
        return None
    try:
        out = llm.complete(LLM_SYS, question, max_tokens=500)
        return json.loads(re.search(r"\{.*\}", out, re.S).group(0)).get("sql")
    except Exception:
        return None


def _brl(x):
    return f"R$ {x:,.0f}"


def explain(plan, df):
    if df.empty:
        return "The query returned no rows."
    it, r = plan.get("intent"), df.iloc[0]
    p = _periods()
    if it == "quarterly_change":
        word = "grew" if r["change"] > 0 else "fell"
        pct = f"{(r['last_revenue'] / r['prev_revenue'] - 1) * 100:+.0f}%"
        return (f"**{r['category']}** {word} the most between {p['prev_label']} and {p['last_label']}: "
                f"{_brl(r['prev_revenue'])} → {_brl(r['last_revenue'])} ({_brl(r['change'])}, {pct}). "
                f"Categories with under R$ 5,000 in {p['prev_label']} are excluded; {p['last_label']} is the last complete quarter.")
    if it == "late_vs_score":
        d = {x["delivery"]: x["avg_score"] for x in df.to_dict("records")}
        return f"Late deliveries average **{d.get('late')}★** versus **{d.get('on time')}★** for on-time deliveries."
    if it == "categories":
        s = f"Top category by revenue: **{r['category']}** ({_brl(r['revenue'])}, {int(r['orders']):,} orders)"
        return s + (f", then **{df.iloc[1]['category']}** ({_brl(df.iloc[1]['revenue'])})." if len(df) > 1 else ".")
    if it == "monthly":
        peak = df.loc[df.revenue.idxmax()]
        return f"Revenue peaked in **{peak['month']}** at {_brl(peak['revenue'])}."
    if it == "states":
        return f"Highest revenue state: **{r['state']}** ({_brl(r['revenue'])})."
    if it == "delivery":
        return f"Highest late-delivery rate in the list: **{r['state']}** ({r['late_pct']}% late, avg {r['avg_delivery_days']} days)."
    if it == "review_score":
        return f"First row: **{r['category']}** with an average review score of {r['avg_score']} over {int(r['reviews']):,} reviews."
    if it == "overview":
        return f"Revenue {_brl(r['revenue'])} from {int(r['orders']):,} orders (average order value R$ {r['avg_order_value']:.2f})."
    if it == "payments":
        return f"Most used payment method by value: **{r['payment_type']}** ({r['pct_of_value']}% of payment value)."
    if it == "status":
        return f"Most common order status: **{r['status']}** ({r['pct']}% of orders)."
    if it == "sellers":
        return f"Top seller (anonymised id {r['seller_id']}, state {r['seller_state']}) earned {_brl(r['revenue'])}."
    return f"Query returned {len(df)} row(s)."


FOREIGN = re.compile(r"(?<![.?!]\s)(?<!^)\b[A-Z][a-z]{2,}")   # capitalised words mid-sentence = named things
ALLOW_CAPS = {"Olist", "Brazil", "Brazilian", "Show", "Top"}


def answer_sql(question, categories_=None):
    base = {"definition": DEFINITION, "data_end": _periods()["data_end"]}
    foreign = [w for w in FOREIGN.findall(question) if w not in ALLOW_CAPS]
    if foreign:   # e.g. "What is Apple's iPhone revenue?" must not be answered with Olist numbers
        return {**base, "answer": f"This looks like a question about something outside the Olist dataset ({', '.join(foreign)}), "
                                  "so I won't answer it from the sales database.", "sql": None, "abstained": True}
    sql = llm_sql(question)
    if sql:
        try:
            df = run_readonly(sql)
            return {**base, "answer": f"Query returned {len(df)} row(s).", "sql": validate_sql(sql), "table": df.to_dict("records"),
                    "sql_source": "claude (validated)", "abstained": False}
        except Exception:
            pass
    plan = build_template(question)
    if plan is None:
        return {**base, "answer": "I can't map this to a supported analytics query. Try revenue, categories, states, delivery times, "
                                  "review scores, payment methods or order status.", "sql": None, "abstained": True}
    df = run_readonly(plan["sql"], plan["params"])
    note = f" (filters applied: {plan['applied']})" if plan["applied"] else ""
    return {**base, "answer": explain(plan, df) + note, "sql": plan["sql"], "params": plan["params"], "title": plan["title"],
            "intent": plan["intent"], "table": df.to_dict("records"), "sql_source": "template", "abstained": False}
