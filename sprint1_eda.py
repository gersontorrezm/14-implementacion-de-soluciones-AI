"""
Sprint 1 – EDA & Baseline
Caso #8: Recomendación de Productos Personalizados
Dataset: Olist Brazilian E-Commerce
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent
# Los CSV pueden estar en la raíz del proyecto o en la subcarpeta data/
if not (DATA_DIR / "olist_orders_dataset.csv").exists() and (DATA_DIR / "data" / "olist_orders_dataset.csv").exists():
    DATA_DIR = DATA_DIR / "data"


# ─────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────

def load_raw_tables():
    orders       = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv", parse_dates=[
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date"])
    customers    = pd.read_csv(DATA_DIR / "olist_customers_dataset.csv")
    items        = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")
    products     = pd.read_csv(DATA_DIR / "olist_products_dataset.csv")
    reviews      = pd.read_csv(DATA_DIR / "olist_order_reviews_dataset.csv")
    payments     = pd.read_csv(DATA_DIR / "olist_order_payments_dataset.csv")
    sellers      = pd.read_csv(DATA_DIR / "olist_sellers_dataset.csv")
    category_map = pd.read_csv(DATA_DIR / "product_category_name_translation.csv")
    return orders, customers, items, products, reviews, payments, sellers, category_map


# ─────────────────────────────────────────────
# 2. MASTER TABLE (~60 metadatos)
# ─────────────────────────────────────────────

def build_master_table(orders, customers, items, products, reviews, payments, sellers, category_map):
    # Pagos agregados por pedido
    pay_agg = (payments.groupby("order_id")
               .agg(total_payment=("payment_value", "sum"),
                    num_installments=("payment_installments", "max"),
                    payment_type=("payment_type", lambda x: x.mode()[0]))
               .reset_index())

    # Reviews: nota promedio por pedido
    rev_agg = (reviews.groupby("order_id")
               .agg(review_score=("review_score", "mean"),
                    review_count=("review_id", "count"))
               .reset_index())

    # Productos + categoría en inglés
    prod = products.merge(category_map, on="product_category_name", how="left")
    prod["category_en"] = prod["product_category_name_english"].fillna(
        prod["product_category_name"].str.replace("_", " ").str.title())

    # Items + productos
    items_prod = items.merge(prod[["product_id", "product_category_name",
                                   "category_en", "product_weight_g",
                                   "product_photos_qty"]], on="product_id", how="left")

    # Unir todo
    master = (orders
              .merge(customers, on="customer_id", how="left")
              .merge(items_prod, on="order_id", how="left")
              .merge(rev_agg, on="order_id", how="left")
              .merge(pay_agg, on="order_id", how="left")
              .merge(sellers[["seller_id", "seller_state"]], on="seller_id", how="left"))

    # Variables derivadas (features mensuales)
    master["delivery_days"] = (
        master["order_delivered_customer_date"] - master["order_purchase_timestamp"]
    ).dt.days
    master["estimated_days"] = (
        master["order_estimated_delivery_date"] - master["order_purchase_timestamp"]
    ).dt.days
    master["delivery_delay"] = master["delivery_days"] - master["estimated_days"]
    master["is_late"] = (master["delivery_delay"] > 0).astype(int)
    master["order_year"]  = master["order_purchase_timestamp"].dt.year
    master["order_month"] = master["order_purchase_timestamp"].dt.month
    master["order_ym"]    = master["order_purchase_timestamp"].dt.to_period("M").astype(str)
    master["item_total"]  = master["price"] + master["freight_value"]
    master["freight_ratio"] = np.where(
        master["price"] > 0, master["freight_value"] / master["price"], np.nan)

    return master


# ─────────────────────────────────────────────
# 3. TABLA CLIENTE-PRODUCTO (base del sistema de recomendación)
# ─────────────────────────────────────────────

def build_user_item_table(master):
    """Una fila = (customer_unique_id, category_en) con métricas agregadas."""
    ui = (master.dropna(subset=["customer_unique_id", "category_en"])
          .groupby(["customer_unique_id", "category_en"])
          .agg(
              purchase_count=("order_id", "nunique"),
              total_spent=("item_total", "sum"),
              avg_review=("review_score", "mean"),
              avg_delivery_days=("delivery_days", "mean"),
          )
          .reset_index())
    return ui


# ─────────────────────────────────────────────
# 4. HISTORIAL POR CLIENTE (para baseline leave-one-out)
# ─────────────────────────────────────────────

def build_customer_history(master):
    hist = (master.dropna(subset=["customer_unique_id", "category_en"])
            .sort_values("order_purchase_timestamp")
            .groupby("customer_unique_id")["category_en"]
            .apply(list)
            .reset_index())
    hist.columns = ["customer_unique_id", "category_history"]
    hist["n_categories"] = hist["category_history"].apply(lambda x: len(set(x)))
    hist["n_orders"]     = hist["category_history"].apply(len)
    return hist


# ─────────────────────────────────────────────
# 5. BASELINE: POPULARIDAD
# ─────────────────────────────────────────────

def compute_popularity_baseline(master, top_k=10):
    """Las top-k categorías más compradas como recomendación universal."""
    pop = (master.groupby("category_en")["order_id"]
           .nunique()
           .sort_values(ascending=False)
           .reset_index()
           .rename(columns={"order_id": "num_orders"}))
    pop["rank"]         = range(1, len(pop) + 1)
    pop["pct_of_total"] = pop["num_orders"] / pop["num_orders"].sum() * 100
    return pop.head(top_k), pop


def precision_recall_at_k(customer_history, popular_items, k=10):
    """
    Leave-one-out evaluation sobre clientes con ≥2 categorías distintas.
    Se retiran las últimas N compras (test) y se recomiendan los top-k populares.
    """
    multi = customer_history[customer_history["n_categories"] >= 2].copy()
    results = []
    popular_set = set(popular_items.head(k)["category_en"].tolist())

    for _, row in multi.iterrows():
        history = row["category_history"]
        unique_cats = list(dict.fromkeys(history))   # orden preservado
        train_cats = set(unique_cats[:-1])
        test_cats  = set(unique_cats[-1:])
        # excluir ya compradas del train del conjunto a recomendar
        candidates = [c for c in popular_items["category_en"] if c not in train_cats]
        recs = set(candidates[:k])
        hits = recs & test_cats
        results.append({
            "precision": len(hits) / k if k > 0 else 0,
            "recall":    len(hits) / len(test_cats) if test_cats else 0,
        })

    df = pd.DataFrame(results)
    return {
        f"Precision@{k}": round(df["precision"].mean(), 4),
        f"Recall@{k}":    round(df["recall"].mean(), 4),
        "n_users_evaluated": len(df),
    }


# ─────────────────────────────────────────────
# 6. BASELINE: CO-COMPRA (item-item collaborative)
# ─────────────────────────────────────────────

def build_copurchase_matrix(master, min_support=20):
    """
    Matriz de co-compra a nivel de categoría:
    cuántos clientes compraron ambas categorías A y B.
    """
    user_cats = (master.dropna(subset=["customer_unique_id", "category_en"])
                 .groupby("customer_unique_id")["category_en"]
                 .apply(set)
                 .reset_index())
    user_cats.columns = ["customer_unique_id", "cats"]

    from itertools import combinations
    from collections import Counter

    pair_counts = Counter()
    for cats in user_cats["cats"]:
        for a, b in combinations(sorted(cats), 2):
            pair_counts[(a, b)] += 1

    records = [{"cat_a": a, "cat_b": b, "support": s}
               for (a, b), s in pair_counts.items() if s >= min_support]
    copurchase = pd.DataFrame(records).sort_values("support", ascending=False)
    return copurchase


# ─────────────────────────────────────────────
# 7. ANÁLISIS DE CALIDAD DE DATOS
# ─────────────────────────────────────────────

def data_quality_report(master):
    total = len(master)
    report = pd.DataFrame({
        "column": master.columns,
        "dtype":  master.dtypes.values,
        "null_count": master.isnull().sum().values,
        "null_pct":   (master.isnull().sum().values / total * 100).round(2),
        "unique":     master.nunique().values,
    }).sort_values("null_pct", ascending=False)
    return report


# ─────────────────────────────────────────────
# 8. MÉTRICAS DE NEGOCIO BASELINE
# ─────────────────────────────────────────────

def business_metrics(master):
    delivered = master[master["order_status"] == "delivered"].copy()
    metrics = {
        "total_orders":           master["order_id"].nunique(),
        "total_customers":        master["customer_unique_id"].nunique(),
        "total_products":         master["product_id"].nunique(),
        "total_categories":       master["category_en"].nunique(),
        "avg_review_score":       round(delivered["review_score"].mean(), 2),
        "avg_items_per_order":    round(master.groupby("order_id")["order_item_id"].count().mean(), 2),
        "avg_order_value_brl":    round(master.groupby("order_id")["item_total"].sum().mean(), 2),
        "avg_delivery_days":      round(delivered["delivery_days"].mean(), 1),
        "late_delivery_rate_pct": round(delivered["is_late"].mean() * 100, 1),
        "repeat_customer_rate":   round(
            (master.groupby("customer_unique_id")["order_id"]
             .nunique() > 1).mean() * 100, 1),
        "avg_categories_per_customer": round(
            master.groupby("customer_unique_id")["category_en"]
            .nunique().mean(), 2),
        "top_state_customers": (master.groupby("customer_state")["customer_unique_id"]
                                .nunique().idxmax()),
    }
    return metrics


if __name__ == "__main__":
    print("Cargando datos...")
    orders, customers, items, products, reviews, payments, sellers, category_map = load_raw_tables()

    print("Construyendo Master Table...")
    master = build_master_table(orders, customers, items, products, reviews,
                                payments, sellers, category_map)
    print(f"  Master Table: {master.shape[0]:,} filas × {master.shape[1]} columnas")

    print("\nMétricas de negocio baseline:")
    bm = business_metrics(master)
    for k, v in bm.items():
        print(f"  {k}: {v}")

    print("\nBaseline de popularidad (top-10 categorías):")
    top10, full_pop = compute_popularity_baseline(master, top_k=10)
    print(top10[["rank", "category_en", "num_orders", "pct_of_total"]].to_string(index=False))

    print("\nEvaluando Precision@10 y Recall@10 (popularity baseline)...")
    hist = build_customer_history(master)
    metrics_k = precision_recall_at_k(hist, full_pop, k=10)
    for k, v in metrics_k.items():
        print(f"  {k}: {v}")

    print("\nSprint 1 completado.")
