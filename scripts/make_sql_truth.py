"""Compute ground-truth numbers for the SQL eval DIRECTLY from the raw Olist CSVs with pandas
(independent of SQLite/templates). Usage: python scripts/make_sql_truth.py path/to/olist_csv_folder"""
import json, sys
from pathlib import Path
import pandas as pd

src = Path(sys.argv[1]); R = lambda n: src / f"olist_{n}_dataset.csv"
o = pd.read_csv(R("orders"), parse_dates=["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"])
i = pd.read_csv(R("order_items")); p = pd.read_csv(R("products")); c = pd.read_csv(R("customers"))
r = pd.read_csv(R("order_reviews")).drop_duplicates("review_id"); pay = pd.read_csv(R("order_payments"))
tr = pd.read_csv(src / "product_category_name_translation.csv")
p = p.merge(tr, how="left"); p["category"] = p.product_category_name_english.fillna(p.product_category_name).fillna("unknown")
ok = o[~o.order_status.isin(["canceled", "unavailable"])]
d = i.merge(ok[["order_id", "customer_id", "order_purchase_timestamp"]]).merge(p[["product_id", "category"]]).merge(c[["customer_id", "customer_state"]])
d["q"] = d.order_purchase_timestamp.dt.to_period("Q").astype(str)
pv = d[d.q.isin(["2018Q1", "2018Q2"])].pivot_table(index="category", columns="q", values="price", aggfunc="sum").fillna(0)
pv["chg"] = pv["2018Q2"] - pv["2018Q1"]
pv = pv[(pv["2018Q1"] >= 5000) & (pv.index != "unknown")]
dl = o[(o.order_status == "delivered") & o.order_delivered_customer_date.notna()].copy()
dl["late"] = dl.order_delivered_customer_date > dl.order_estimated_delivery_date
rv = r.merge(dl[["order_id", "late"]])
truth = {
 "total_revenue": round(float(d.price.sum()), 2), "total_orders": int(d.order_id.nunique()),
 "top_category": d.groupby("category").price.sum().idxmax(),
 "biggest_drop_category": pv.chg.idxmin(), "biggest_drop_amount": round(float(pv.chg.min()), 2),
 "top_state": d.groupby("customer_state").price.sum().idxmax(),
 "top_payment_type": pay.groupby("payment_type").payment_value.sum().idxmax(),
 "avg_score_late": round(float(rv[rv.late].review_score.mean()), 2),
 "avg_score_on_time": round(float(rv[~rv.late].review_score.mean()), 2),
 "top_seller_revenue": round(float(d.merge(i[["order_id", "order_item_id", "seller_id"]]).groupby("seller_id").price.sum().max()), 2),
}
Path(__file__).resolve().parent.parent.joinpath("eval", "sql_truth.json").write_text(json.dumps(truth, indent=1))
print(json.dumps(truth, indent=1))
