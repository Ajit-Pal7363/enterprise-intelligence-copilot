"""Pull simple entities (year, state, category, top-N) out of a question."""
import re

STATES = {"AC","AL","AP","AM","BA","CE","DF","ES","GO","MA","MT","MS","MG","PA","PB","PR","PE","PI","RJ","RN","RS","RO","RR","SC","SP","SE","TO"}
STATE_NAMES = {"sao paulo": "SP", "são paulo": "SP", "rio de janeiro": "RJ", "minas gerais": "MG", "bahia": "BA",
               "parana": "PR", "paraná": "PR", "rio grande do sul": "RS", "santa catarina": "SC"}
SYN = {"electronics": "electronics", "furniture": "furniture_decor", "beauty": "health_beauty", "health": "health_beauty",
       "sports": "sports_leisure", "sport": "sports_leisure", "toys": "toys", "toy": "toys", "computers": "computers",
       "phones": "telephony", "phone": "telephony", "perfume": "perfumery", "watches": "watches_gifts",
       "garden": "garden_tools", "cars": "auto", "auto": "auto", "books": "books_general_interest",
       "baby": "baby", "pet": "pet_shop", "pets": "pet_shop", "fashion": "fashion_bags_accessories",
       "food": "food", "appliances": "home_appliances", "bed": "bed_bath_table", "bath": "bed_bath_table",
       "housewares": "housewares", "stationery": "stationery", "music": "musical_instruments", "luggage": "luggage_accessories"}


def find_category(q: str, categories):
    toks = re.findall(r"[a-z]+", q.lower())
    cats = set(categories)
    for t in toks:
        for key in (t, t.rstrip("s")):
            if key in SYN and SYN[key] in cats:
                return SYN[key]
    best, score = None, 0
    for c in cats:
        parts = [p for p in c.split("_") if len(p) > 3]
        s = sum(1 for p in parts if p in toks or p.rstrip("s") in toks)
        if s > score:
            best, score = c, s
    return best if score else None


def parse(q: str, categories=()):
    year = re.search(r"\b(2016|2017|2018)\b", q)
    state = next((v for k, v in STATE_NAMES.items() if k in q.lower()), None)
    if not state:
        state = next((t for t in re.findall(r"\b[A-Z]{2}\b", q) if t in STATES), None)
    n = re.search(r"\b(?:top|bottom|worst|best|lowest|highest)\s+(\d{1,2})\b", q, re.I)
    return {"year": year.group(1) if year else None, "state": state,
            "category": find_category(q, categories) if categories else None,
            "topn": min(int(n.group(1)), 50) if n else None,
            "ascending": bool(re.search(r"worst|lowest|least|bottom|fewest|smallest", q, re.I))}
