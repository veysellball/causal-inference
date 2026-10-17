from pathlib import Path
import re

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

START, END = "2017-01-01", "2018-08-31"     # same window as Project 3 (MATURE_UNTIL)

TABLES = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


def connect_raw() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for name, file in TABLES.items():
        path = (RAW / file).as_posix()
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM read_csv_auto('{path}')")
    return con


ANALYSIS_SQL = """
WITH geo AS (
    -- several rows per zip prefix: collapse to one point (grain!), drop points outside Brazil
    SELECT geolocation_zip_code_prefix AS zip,
           MEDIAN(geolocation_lat) AS lat,
           MEDIAN(geolocation_lng) AS lng
    FROM geolocation
    WHERE geolocation_lat BETWEEN -34 AND 6 AND geolocation_lng BETWEEN -74 AND -34
    GROUP BY 1
),
items AS (
    SELECT oi.order_id,
           COUNT(DISTINCT oi.seller_id) AS n_sellers,
           ANY_VALUE(oi.seller_id)      AS seller_id,
           COUNT(*)                     AS n_items,
           SUM(oi.price)                AS price,
           SUM(oi.freight_value)        AS freight,
           SUM(p.product_weight_g)      AS weight_g,
           arg_max(COALESCE(t.product_category_name_english,
                            p.product_category_name, 'unknown'), oi.price) AS category
    FROM order_items oi
    LEFT JOIN products p             ON oi.product_id = p.product_id
    LEFT JOIN category_translation t ON p.product_category_name = t.product_category_name
    GROUP BY 1
),
latest_review AS (
    SELECT order_id,
           arg_max(review_score, review_answer_timestamp)         AS review_score,
           arg_max(review_creation_date, review_answer_timestamp) AS review_created
    FROM reviews
    GROUP BY 1
),
delivered AS (
    SELECT
        o.order_id,
        o.order_purchase_timestamp                                AS purchase_ts,
        CAST(o.order_purchase_timestamp AS DATE)                  AS purchase_date,
        CAST(o.order_delivered_customer_date AS DATE)             AS delivered_date,
        DATE_DIFF('day', CAST(o.order_purchase_timestamp AS DATE),
                         CAST(o.order_estimated_delivery_date AS DATE))   AS promised_days,
        DATE_DIFF('day', CAST(o.order_purchase_timestamp AS DATE),
                         CAST(o.order_delivered_customer_date AS DATE))   AS actual_days,
        -- > 0 means late; day-level comparison (Project 1 fix)
        DATE_DIFF('day', CAST(o.order_estimated_delivery_date AS DATE),
                         CAST(o.order_delivered_customer_date AS DATE))   AS days_late,
        c.customer_state,
        c.customer_zip_code_prefix  AS customer_zip,
        s.seller_state,
        s.seller_zip_code_prefix    AS seller_zip,
        i.seller_id, i.n_sellers, i.n_items, i.price, i.freight, i.weight_g, i.category,
        r.review_score,
        r.review_created
    FROM orders o
    JOIN customers c          ON o.customer_id = c.customer_id
    JOIN items i              ON o.order_id = i.order_id
    JOIN sellers s            ON i.seller_id = s.seller_id
    LEFT JOIN latest_review r ON o.order_id = r.order_id
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
),
seller_history AS (
    -- what was known about the seller from orders placed BEFORE this one
    SELECT order_id,
           COUNT(*) OVER w                          AS seller_prior_orders,
           AVG(CAST(days_late > 0 AS INTEGER)) OVER w AS seller_prior_late_rate,
           AVG(review_score) OVER w                 AS seller_prior_review
    FROM delivered
    WINDOW w AS (PARTITION BY seller_id ORDER BY purchase_ts
                 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
)
SELECT
    d.*,
    h.seller_prior_orders, h.seller_prior_late_rate, h.seller_prior_review,
    d.days_late > 0                    AS is_late,
    d.review_score >= 4                AS good_review,
    d.customer_state = d.seller_state  AS same_state,
    2 * 6371 * ASIN(SQRT(
        POWER(SIN(RADIANS(gs.lat - gc.lat) / 2), 2)
        + COS(RADIANS(gc.lat)) * COS(RADIANS(gs.lat))
          * POWER(SIN(RADIANS(gs.lng - gc.lng) / 2), 2)
    ))                                 AS distance_km
FROM delivered d
JOIN seller_history h ON d.order_id = h.order_id
LEFT JOIN geo gc ON d.customer_zip = gc.zip
LEFT JOIN geo gs ON d.seller_zip = gs.zip
WHERE d.purchase_date BETWEEN DATE '{start}' AND DATE '{end}'
"""


def build_raw_table(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(ANALYSIS_SQL.format(start=START, end=END)).df()
    for col in ["purchase_date", "delivered_date", "review_created"]:
        df[col] = pd.to_datetime(df[col])
    return df


REGIONS = {
    "N":  ["AC", "AM", "AP", "PA", "RO", "RR", "TO"],
    "NE": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
    "CO": ["DF", "GO", "MS", "MT"],
    "SE": ["ES", "MG", "RJ", "SP"],
    "S":  ["PR", "RS", "SC"],
}
STATE_TO_REGION = {s: r for r, states in REGIONS.items() for s in states}

NUMERIC_CONFOUNDERS = [
    "log_distance", "same_state", "promised_days", "log_freight", "freight_share",
    "log_price", "log_weight", "n_items",
    "seller_prior_late_rate", "seller_prior_review", "log_seller_prior_orders", "seller_is_new",
]


def _safe(name: str) -> str:
    return re.sub(r"\W+", "_", name).strip("_").lower()


def prepare(raw: pd.DataFrame, top_categories: int = 15) -> tuple[pd.DataFrame, list[str]]:
    """Filter to the analysis population and encode confounders.
    Returns the table and the full list of adjustment columns."""
    df = raw[(raw["n_sellers"] == 1)
             & raw["review_score"].notna()
             & raw["distance_km"].notna()].copy()

    df["is_late"] = df["is_late"].astype(bool)
    df["good_review"] = df["good_review"].astype(int)
    df["seller_is_new"] = (df["seller_prior_orders"] == 0).astype(int)
    for col in ["seller_prior_late_rate", "seller_prior_review"]:
        df[col] = df[col].fillna(df[col].mean())      # new sellers: neutral value + the flag above
    df["weight_g"] = df["weight_g"].fillna(df["weight_g"].median())

    df["log_distance"] = np.log1p(df["distance_km"])
    df["log_freight"] = np.log1p(df["freight"])
    df["log_price"] = np.log1p(df["price"])
    df["log_weight"] = np.log1p(df["weight_g"])
    df["log_seller_prior_orders"] = np.log1p(df["seller_prior_orders"])
    df["freight_share"] = df["freight"] / (df["price"] + df["freight"])
    df["same_state"] = df["same_state"].astype(int)

    df["region"] = df["customer_state"].map(STATE_TO_REGION)
    top = df["category"].value_counts().index[:top_categories]
    df["category_grp"] = df["category"].where(df["category"].isin(top), "other")
    df["purchase_month"] = df["purchase_date"].dt.strftime("%Y_%m")

    dummies = pd.get_dummies(df[["region", "category_grp", "purchase_month"]],
                             prefix=["region", "cat", "month"], drop_first=True, dtype=int)
    dummies.columns = [_safe(c) for c in dummies.columns]
    df = pd.concat([df, dummies], axis=1)

    confounders = NUMERIC_CONFOUNDERS + list(dummies.columns)
    return df, confounders


def build_dot(confounders: list[str], treatment: str, outcome: str) -> str:
    """Backdoor-adjustment graph for DoWhy: every confounder → T and → Y, plus T → Y."""
    edges = [f"{c} -> {treatment}; {c} -> {outcome};" for c in confounders]
    return "digraph {" + " ".join(edges) + f" {treatment} -> {outcome};" + "}"



def smd(df: pd.DataFrame, cols: list[str], treat: str, weights=None) -> pd.Series:
    """Standardized mean difference (treated − control) / pooled SD, optionally weighted."""
    w = np.ones(len(df)) if weights is None else np.asarray(weights)
    t = df[treat].to_numpy(bool)
    out = {}
    for c in cols:
        x = df[c].to_numpy(float)
        m1, m0 = np.average(x[t], weights=w[t]), np.average(x[~t], weights=w[~t])
        v1, v0 = x[t].var(), x[~t].var()            # unweighted SDs as the common yardstick
        out[c] = (m1 - m0) / np.sqrt((v1 + v0) / 2) if v1 + v0 > 0 else 0.0
    return pd.Series(out)