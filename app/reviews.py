"""Customer Intelligence over Olist reviews (Portuguese): rule-based complaint themes with counts and verbatim evidence.
Themes overlap (a review can match several). Keyword rules, not a trained model."""
import re
import pandas as pd
from .db import connect_ro

THEMES = {
    "Late delivery / delay": r"atras|demor|prazo|muito tempo|ainda n[aã]o (chegou|recebi)",
    "Product never received": r"n[aã]o (recebi|chegou|foi entregue|entregaram)|nunca chegou|n[aã]o veio",
    "Defective / not working": r"defeit|n[aã]o funciona|quebr|danific|estragad|parou de funcionar",
    "Wrong or incomplete item": r"errad|diferente|faltand|falt(a|ou)|incomplet|veio (apenas|s[oó]|somente)|apenas um",
    "Poor quality / not as described": r"qualidade|p[eé]ssim|ruim|fraco|propaganda enganosa",
    "Refund / seller support": r"reembols|devol|estorno|cancel|atendimento|sem resposta|contato",
}


def analyze_reviews(category=None, evidence=2):
    cat_sql = "AND p.category = ?" if category else ""
    sql = ("SELECT r.review_id, r.score, r.comment, "
           "(SELECT p.category FROM order_items oi JOIN products p ON p.product_id = oi.product_id WHERE oi.order_id = r.order_id LIMIT 1) AS category "
           "FROM reviews r")
    if category:
        sql = ("SELECT DISTINCT r.review_id, r.score, r.comment, p.category AS category FROM reviews r "
               "JOIN order_items oi ON oi.order_id = r.order_id JOIN products p ON p.product_id = oi.product_id "
               f"WHERE 1=1 {cat_sql}")
    with connect_ro() as con:
        allr = pd.read_sql(sql, con, params=[category] if category else [])
    if allr.empty:
        return {"scope": category or "all categories", "total_reviews": 0, "themes": []}
    neg = allr[(allr.score <= 2) & allr.comment.fillna("").str.strip().ne("")].copy()
    low = neg.comment.str.lower()
    themes = []
    for name, rx in THEMES.items():
        m = low.str.contains(rx, regex=True)
        ex = neg[m].assign(n=lambda d: d.comment.str.len()).query("40 <= n <= 220").sort_values("n").head(evidence)
        themes.append({"theme": name, "reviews": int(m.sum()), "share_pct": round(100 * float(m.mean()), 1) if len(neg) else 0.0,
                       "evidence": [{"review_id": int(r.review_id), "score": int(r.score), "category": r.category, "text": r.comment}
                                    for r in ex.itertuples()]})
    themes.sort(key=lambda t: -t["reviews"])
    return {"scope": category or "all categories", "total_reviews": int(len(allr)), "avg_score": round(float(allr.score.mean()), 2),
            "pct_negative": round(100 * float((allr.score <= 2).mean()), 1), "negative_with_comment": int(len(neg)), "themes": themes}


def summarize(res):
    if not res["themes"]:
        return "No reviews found for this scope."
    t = res["themes"][0]
    return (f"For **{res['scope']}**: {res['total_reviews']:,} reviews, average score {res['avg_score']}/5, {res['pct_negative']}% negative. "
            f"Among {res['negative_with_comment']:,} negative (1–2★) reviews with a comment, the most frequent complaint theme is "
            f"**{t['theme']}** ({t['reviews']:,} reviews, {t['share_pct']}%). Themes overlap; evidence quotes are the original Portuguese text.")
