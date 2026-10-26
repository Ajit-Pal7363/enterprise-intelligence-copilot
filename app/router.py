"""Query router: weighted keyword scoring (transparent + testable). Ties -> docs > reviews > sql.
route(q) -> (route, scores)."""
import re

P = {
    "docs": [r"grievance", r"consumer", r"e-?commerce (rules?|entity|entities)", r"rule \d+", r"\brules?\b", r"nodal",
             r"marketplace", r"inventory e-?commerce", r"unfair trade", r"10-?k", r"annual report", r"risk factors?",
             r"\bamazon\b", r"\baws\b", r"sec filing", r"item 1a", r"fiscal year", r"\blegal(ly)?\b", r"\blaw\b", r"\bact\b",
             r"regulation", r"compliance", r"headquarter", r"force majeure", r"excus", r"refund polic", r"polic(y|ies)",
             r"redress", r"platforms?\b", r"cancell?ation (charge|fee)|charge fees?", r"\bbuyer", r"duties of", r"liabilit",
             r"seasonality", r"cybersecurity", r"employees?\b.*\b(amazon|company)", r"\bceo\b", r"intellectual property"],
    "reviews": [(r"complain", 3), (r"negative review|bad review", 3), (r"what (do )?customers (say|think|dislike|hate)", 3),
                (r"common (issue|problem)", 3), (r"why (are|do) customers", 3), (r"sentiment|review comments?", 3),
                (r"root cause|pain points?", 3), (r"(worst|best|lowest|highest).{0,40}(review|rating|score)", 8)],
    "sql": [r"revenue", r"\bsales\b", r"gmv", r"\borders?\b", r"\baov\b", r"average order", r"\bquarter", r"monthly|per month|by month",
            r"by (state|category|seller|payment)", r"categor(y|ies)", r"\bsellers?\b", r"payment (type|method)|installments|boleto",
            r"deliver(y|ies|ed)|\blate\b", r"order status", r"review score|average rating|rating by", r"\bolist\b",
            r"\btop \d+", r"\b(drop|dropped|fell|fall|declin)", r"gira", r"\bstates?\b"],
}


def score(q: str):
    ql, out = q.lower(), {}
    for k, ps in P.items():
        tot = 0
        for p in ps:
            pat, w = p if isinstance(p, tuple) else (p, 3)
            if re.search(pat, ql):
                tot += w
        out[k] = tot
    return out


def route(q: str):
    s = score(q)
    best = max(s.values())
    if best == 0:
        return "docs", s
    for k in ("docs", "reviews", "sql"):
        if s[k] == best:
            return k, s
