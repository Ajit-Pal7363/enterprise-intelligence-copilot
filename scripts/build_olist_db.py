"""Build data/olist.db from the raw Olist CSVs (Kaggle: olistbr/brazilian-ecommerce).
Usage: python scripts/build_olist_db.py /path/to/folder_with_csvs
Drops geolocation/customer identifiers to keep the DB small and privacy-friendly."""
import sqlite3, sys
from pathlib import Path
import pandas as pd

raw = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw_olist")
out = Path(__file__).resolve().parent.parent / "data" / "olist.db"
out.unlink(missing_ok=True)
rd = lambda n, **k: pd.read_csv(raw / n, **k)

orders = rd("olist_orders_dataset.csv")
cust = rd("olist_customers_dataset.csv")[["customer_id", "customer_state", "customer_city"]]
o = orders.merge(cust, on="customer_id", how="left")
for c in ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]:
    o[c] = pd.to_datetime(o[c])
o["purchase_date"] = o["order_purchase_timestamp"].dt.strftime("%Y-%m-%d")
o["purchase_month"] = o["order_purchase_timestamp"].dt.strftime("%Y-%m")
o["delivered_date"] = o["order_delivered_customer_date"].dt.strftime("%Y-%m-%d")
o["estimated_date"] = o["order_estimated_delivery_date"].dt.strftime("%Y-%m-%d")
o["delivery_days"] = (o["order_delivered_customer_date"] - o["order_purchase_timestamp"]).dt.total_seconds() / 86400
o["delivery_days"] = o["delivery_days"].round(1)
o["is_late"] = (o["order_delivered_customer_date"] > o["order_estimated_delivery_date"]).astype("Int64")
o.loc[o["order_delivered_customer_date"].isna(), "is_late"] = pd.NA
customers = o[["customer_id", "customer_state"]].drop_duplicates("customer_id").rename(columns={"customer_state": "state"})
o = o.rename(columns={"order_status": "status"})[
    ["order_id", "customer_id", "customer_state", "status", "purchase_date", "purchase_month", "delivered_date",
     "estimated_date", "delivery_days", "is_late"]]

items = rd("olist_order_items_dataset.csv")[["order_id", "product_id", "seller_id", "price", "freight_value"]]
prod = rd("olist_products_dataset.csv")[["product_id", "product_category_name"]]
tr = rd("product_category_name_translation.csv")
prod = prod.merge(tr, on="product_category_name", how="left")
prod["category"] = prod["product_category_name_english"].fillna(prod["product_category_name"]).fillna("unknown")
prod = prod[["product_id", "category"]]
pay = rd("olist_order_payments_dataset.csv")[["order_id", "payment_type", "payment_installments", "payment_value"]]
rev = rd("olist_order_reviews_dataset.csv")
rev = rev.rename(columns={"review_score": "score", "review_comment_message": "comment", "review_creation_date": "created_date"})
rev["created_date"] = pd.to_datetime(rev["created_date"]).dt.strftime("%Y-%m-%d")
rev = rev.drop_duplicates("review_id")[["review_id", "order_id", "score", "comment", "created_date"]]
sell = rd("olist_sellers_dataset.csv")[["seller_id", "seller_state", "seller_city"]]

# compact integer ids (the hex ids are anonymised hashes anyway)
def remap(col, *frames):
    allv = pd.concat([f[col] for f in frames if col in f]).dropna().unique()
    m = {v: i + 1 for i, v in enumerate(allv)}
    for f in frames:
        if col in f:
            f[col] = f[col].map(m).astype("Int64")
for col, frs in [("customer_id", (o, customers)), ("order_id", (o, items, pay, rev)), ("product_id", (items, prod)),
                 ("seller_id", (items, sell)), ("review_id", (rev,))]:
    remap(col, *frs)

con = sqlite3.connect(out)
for name, df in [("orders", o), ("order_items", items), ("products", prod), ("payments", pay),
                 ("reviews", rev), ("sellers", sell), ("customers", customers)]:
    df.to_sql(name, con, index=False)
for ddl in ["CREATE INDEX i1 ON order_items(order_id)", "CREATE INDEX i2 ON order_items(product_id)",
            "CREATE INDEX i3 ON reviews(order_id)", "CREATE INDEX i4 ON payments(order_id)",
            "CREATE INDEX i5 ON orders(purchase_date)", "CREATE UNIQUE INDEX i6 ON orders(order_id)",
            "CREATE INDEX i7 ON products(product_id)"]:
    con.execute(ddl)
con.execute("CREATE INDEX i8 ON customers(customer_id)")
con.commit(); con.execute("VACUUM"); con.close()
import gzip, shutil
with open(out, "rb") as f, gzip.open(str(out) + ".gz", "wb", compresslevel=9) as g:
    shutil.copyfileobj(f, g)
print("also wrote", str(out) + ".gz", round(Path(str(out) + ".gz").stat().st_size / 1e6, 1), "MB")
print("built", out, round(out.stat().st_size / 1e6, 1), "MB")
