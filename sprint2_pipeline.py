"""
Sprint 2 – Feature Engineering & Pipeline Reproducible
Caso #8: Recomendación de Productos Personalizados
Dataset: Olist Brazilian E-Commerce
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

from sprint1_eda import (
    load_raw_tables,
    build_master_table,
    build_customer_history,
    compute_popularity_baseline,
    precision_recall_at_k,
    business_metrics,
)


# ─────────────────────────────────────────────
# 1. LIMPIEZA Y ESTANDARIZACIÓN
# ─────────────────────────────────────────────

def clean_master_table(master: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica limpieza estándar:
    - Elimina pedidos sin categoría (no aportan al sistema de recomendación)
    - Imputa delivery_days con la mediana por estado vendedor
    - Imputa review_score con la mediana global
    - Elimina outliers de precio (IQR × 3)
    - Filtra pedidos cancelados
    """
    df = master.copy()

    # Eliminar filas sin categoría (esenciales para recomendación)
    before = len(df)
    df = df.dropna(subset=["category_en", "customer_unique_id"])
    rows_dropped_no_cat = before - len(df)

    # Solo pedidos entregados o en tránsito (excluir cancelados y unavailable)
    df = df[~df["order_status"].isin(["canceled", "unavailable", "created"])]

    # Imputar delivery_days con mediana por estado del vendedor
    median_delivery = df.groupby("seller_state")["delivery_days"].transform("median")
    df["delivery_days"] = df["delivery_days"].fillna(median_delivery)
    df["delivery_days"] = df["delivery_days"].fillna(df["delivery_days"].median())

    # Imputar review_score con mediana global
    df["review_score"] = df["review_score"].fillna(df["review_score"].median())

    # Eliminar outliers de precio (precio > Q3 + 3*IQR)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    q3 = df["price"].quantile(0.75)
    iqr = df["price"].quantile(0.75) - df["price"].quantile(0.25)
    price_cap = q3 + 3 * iqr
    before_outlier = len(df)
    df = df[df["price"] <= price_cap]
    rows_dropped_price = before_outlier - len(df)

    df["_cleaning_log"] = f"Dropped no-category: {rows_dropped_no_cat} | price outliers: {rows_dropped_price}"
    return df


def get_cleaning_report(master: pd.DataFrame) -> pd.DataFrame:
    """Genera informe comparativo antes/después de limpieza."""
    clean = clean_master_table(master)
    report = pd.DataFrame({
        "Métrica": [
            "Total filas",
            "Pedidos únicos",
            "Clientes únicos",
            "% nulos delivery_days",
            "% nulos review_score",
            "% nulos category_en",
            "Precio máximo (BRL)",
            "Precio mediana (BRL)",
        ],
        "Antes": [
            len(master),
            master["order_id"].nunique(),
            master["customer_unique_id"].nunique(),
            f"{master['delivery_days'].isnull().mean()*100:.1f}%",
            f"{master['review_score'].isnull().mean()*100:.1f}%",
            f"{master['category_en'].isnull().mean()*100:.1f}%",
            f"{master['price'].max():.0f}",
            f"{master['price'].median():.0f}",
        ],
        "Después": [
            len(clean),
            clean["order_id"].nunique(),
            clean["customer_unique_id"].nunique(),
            f"{clean['delivery_days'].isnull().mean()*100:.1f}%",
            f"{clean['review_score'].isnull().mean()*100:.1f}%",
            f"{clean['category_en'].isnull().mean()*100:.1f}%",
            f"{clean['price'].max():.0f}",
            f"{clean['price'].median():.0f}",
        ],
    })
    return report, clean


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def build_rfm_features(master: pd.DataFrame) -> pd.DataFrame:
    """
    RFM extendido por cliente:
    R = Recency (días desde última compra)
    F = Frequency (pedidos totales)
    M = Monetary (gasto total BRL)
    + categoría preferida, diversidad, satisfacción promedio, etc.
    """
    ref_date = master["order_purchase_timestamp"].max()

    rfm = (master.groupby("customer_unique_id")
           .agg(
               recency=("order_purchase_timestamp", lambda x: (ref_date - x.max()).days),
               frequency=("order_id", "nunique"),
               monetary=("item_total", "sum"),
               avg_ticket=("item_total", "mean"),
               avg_review=("review_score", "mean"),
               category_diversity=("category_en", "nunique"),
               total_items=("order_item_id", "count"),
               avg_delivery_days=("delivery_days", "mean"),
               late_rate=("is_late", "mean"),
               preferred_state=("customer_state", lambda x: x.mode()[0] if len(x) > 0 else ""),
               preferred_category=("category_en", lambda x: x.mode()[0] if len(x) > 0 else ""),
               last_category=("category_en", "last"),
               n_distinct_payments=("payment_type", "nunique"),
           )
           .reset_index())

    # Segmento RFM normalizado (1-5 por cada dimensión)
    def safe_qcut(series, q, labels):
        try:
            return pd.qcut(series, q=q, labels=labels, duplicates="drop").astype(float)
        except Exception:
            # Si no hay suficientes bins únicos, usar ntile manual
            pct = series.rank(pct=True)
            return pd.cut(pct, bins=q, labels=labels, include_lowest=True).astype(float)

    rfm["R_score"] = safe_qcut(rfm["recency"],   5, [5,4,3,2,1])
    rfm["F_score"] = safe_qcut(rfm["frequency"], 5, [1,2,3,4,5])
    rfm["M_score"] = safe_qcut(rfm["monetary"],  5, [1,2,3,4,5])
    rfm["RFM_score"] = rfm[["R_score","F_score","M_score"]].mean(axis=1).round(1)

    rfm["segment"] = pd.cut(
        rfm["RFM_score"],
        bins=[0, 1.5, 2.5, 3.5, 4.5, 5.1],
        labels=["Perdidos", "En riesgo", "Regulares", "Leales", "Campeones"],
    )
    return rfm


def build_category_features(master: pd.DataFrame) -> pd.DataFrame:
    """Features a nivel de categoría para content-based filtering."""
    cat_feat = (master.groupby("category_en")
                .agg(
                    total_orders=("order_id", "nunique"),
                    total_customers=("customer_unique_id", "nunique"),
                    avg_price=("price", "mean"),
                    avg_review=("review_score", "mean"),
                    avg_delivery=("delivery_days", "mean"),
                    late_rate=("is_late", "mean"),
                    total_revenue=("item_total", "sum"),
                    avg_freight_ratio=("freight_ratio", "mean"),
                )
                .reset_index())

    cat_feat["repeat_rate"] = (
        master[master.groupby(["customer_unique_id","category_en"])
               ["order_id"].transform("nunique") > 1]
        .groupby("category_en")["customer_unique_id"].nunique()
        / cat_feat.set_index("category_en")["total_customers"]
    ).fillna(0).values

    cat_feat["popularity_rank"] = cat_feat["total_orders"].rank(ascending=False).astype(int)
    return cat_feat


# ─────────────────────────────────────────────
# 3. MATRIZ USUARIO-CATEGORÍA (base del modelo)
# ─────────────────────────────────────────────

def build_interaction_matrix(master: pd.DataFrame):
    """Construye la matriz dispersa usuario × categoría (conteo de compras)."""
    ui = (master.dropna(subset=["customer_unique_id", "category_en"])
          .groupby(["customer_unique_id", "category_en"])["order_id"]
          .nunique()
          .reset_index())
    ui.columns = ["user", "category", "interactions"]

    users = ui["user"].unique()
    cats  = ui["category"].unique()
    u2i   = {u: i for i, u in enumerate(users)}
    c2i   = {c: i for i, c in enumerate(cats)}

    rows = ui["user"].map(u2i).values
    cols = ui["category"].map(c2i).values
    data = ui["interactions"].values

    matrix = sp.csr_matrix((data, (rows, cols)), shape=(len(users), len(cats)))
    return matrix, users, cats, u2i, c2i


def matrix_stats(matrix, users, cats) -> dict:
    """Estadísticas de la matriz de interacciones."""
    density = matrix.nnz / (matrix.shape[0] * matrix.shape[1]) * 100
    interactions_per_user = np.diff(matrix.indptr)
    interactions_per_cat  = np.diff(matrix.tocsc().indptr)
    return {
        "n_users": len(users),
        "n_categories": len(cats),
        "n_interactions": matrix.nnz,
        "sparsity_pct": round(100 - density, 3),
        "density_pct": round(density, 4),
        "avg_cats_per_user": round(interactions_per_user.mean(), 3),
        "max_cats_per_user": int(interactions_per_user.max()),
        "avg_users_per_cat": round(interactions_per_cat.mean(), 1),
    }


# ─────────────────────────────────────────────
# 4. MODELO SVD (filtrado colaborativo item-item)
# ─────────────────────────────────────────────

def train_svd_model(matrix: sp.csr_matrix, n_components: int = 15):
    """
    Aplica TruncatedSVD a la matriz usuario×categoría.
    Extrae embeddings de categorías para calcular similaridad coseno.
    """
    svd = TruncatedSVD(n_components=n_components, random_state=42, n_iter=10)
    svd.fit(matrix)
    # Factores de categoría: columnas del espacio latente
    cat_factors  = svd.components_.T   # (n_cats, n_components)
    user_factors = svd.transform(matrix)  # (n_users, n_components)
    return svd, cat_factors, user_factors


def compute_category_similarity(cat_factors: np.ndarray, cats) -> pd.DataFrame:
    """Matriz de similitud coseno entre categorías en espacio latente."""
    sim = cosine_similarity(cat_factors)
    return pd.DataFrame(sim, index=cats, columns=cats)


def recommend_svd(purchased_cats: list, cat_sim_df: pd.DataFrame,
                  top_k: int = 10) -> list:
    """
    Recomendación ítem-ítem basada en similitud SVD.
    Para cada categoría comprada, acumula scores de categorías similares.
    Excluye las ya compradas.
    """
    if not purchased_cats or cat_sim_df.empty:
        return []

    purchased_set = set(purchased_cats)
    scores = pd.Series(0.0, index=cat_sim_df.columns, dtype=float)

    for cat in purchased_cats:
        if cat in cat_sim_df.index:
            scores += cat_sim_df.loc[cat]

    scores = scores.drop(labels=[c for c in purchased_set if c in scores.index], errors="ignore")
    return scores.nlargest(top_k).index.tolist()


def evaluate_svd_model(customer_history: pd.DataFrame, cat_sim_df: pd.DataFrame,
                       popular_fallback: pd.DataFrame, k: int = 10) -> dict:
    """Leave-one-out evaluation del modelo SVD."""
    multi = customer_history[customer_history["n_categories"] >= 2]
    results = []

    pop_list = popular_fallback["category_en"].tolist()

    for _, row in multi.iterrows():
        history   = row["category_history"]
        unique    = list(dict.fromkeys(history))
        train_cats = unique[:-1]
        test_cat   = unique[-1]

        recs = recommend_svd(train_cats, cat_sim_df, top_k=k)
        if not recs:
            recs = [c for c in pop_list if c not in set(train_cats)][:k]

        hit = 1 if test_cat in recs else 0
        results.append({"precision": hit / k, "recall": float(hit)})

    df = pd.DataFrame(results)
    return {
        f"Precision@{k}": round(df["precision"].mean(), 4),
        f"Recall@{k}":    round(df["recall"].mean(), 4),
        "n_users":        len(df),
    }


# ─────────────────────────────────────────────
# 5. COMPARACIÓN DE MODELOS (todas las métricas)
# ─────────────────────────────────────────────

def compare_models(customer_history: pd.DataFrame, cat_sim_df: pd.DataFrame,
                   popular_fallback: pd.DataFrame) -> pd.DataFrame:
    """Compara baseline de popularidad vs SVD en todos los K."""
    rows = []
    for k in [5, 10, 20]:
        base = precision_recall_at_k(customer_history, popular_fallback, k=k)
        svd  = evaluate_svd_model(customer_history, cat_sim_df, popular_fallback, k=k)
        def delta(a, b):
            pct = (b / a - 1) * 100 if a > 0 else 0
            return f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%"
        rows.append({
            "K": k,
            "Precision_baseline": base[f"Precision@{k}"],
            "Precision_SVD":      svd[f"Precision@{k}"],
            "Recall_baseline":    base[f"Recall@{k}"],
            "Recall_SVD":         svd[f"Recall@{k}"],
            "Delta_Precision":    delta(base[f"Precision@{k}"], svd[f"Precision@{k}"]),
            "Delta_Recall":       delta(base[f"Recall@{k}"], svd[f"Recall@{k}"]),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 6. DIVISIÓN TEMPORAL (train / val / backtest)
# ─────────────────────────────────────────────

def build_temporal_splits(master: pd.DataFrame) -> dict:
    """
    División cronológica del dataset:
    - Train:    2016-10 → 2017-09  (12 meses)
    - Val:      2017-10 → 2018-03  ( 6 meses)
    - Backtest: 2018-04 → 2018-10  ( 7 meses)
    """
    df = master.dropna(subset=["order_ym"])
    months = sorted(df["order_ym"].unique())

    train_end    = "2017-09"
    val_end      = "2018-03"

    train    = df[df["order_ym"] <= train_end]
    val      = df[(df["order_ym"] > train_end) & (df["order_ym"] <= val_end)]
    backtest = df[df["order_ym"] > val_end]

    summary = pd.DataFrame({
        "Periodo":  ["Train", "Validación", "Backtest"],
        "Desde":    [months[0], "2017-10", "2018-04"],
        "Hasta":    ["2017-09", "2018-03", months[-1]],
        "Meses":    [
            len([m for m in months if m <= train_end]),
            len([m for m in months if train_end < m <= val_end]),
            len([m for m in months if m > val_end]),
        ],
        "Pedidos":  [train["order_id"].nunique(),
                     val["order_id"].nunique(),
                     backtest["order_id"].nunique()],
        "Clientes": [train["customer_unique_id"].nunique(),
                     val["customer_unique_id"].nunique(),
                     backtest["customer_unique_id"].nunique()],
    })
    return {"train": train, "val": val, "backtest": backtest, "summary": summary}


def evaluate_on_split(split_data: dict, cat_sim_df: pd.DataFrame,
                      popular_fallback: pd.DataFrame) -> pd.DataFrame:
    """
    Evalúa el modelo SVD entrenado en Train sobre Val y Backtest.
    Métrica: Recall@10 (hit rate) - qué % de los clientes reciben
    al menos una recomendación acertada en su próxima compra.
    """
    results = []
    for period_name in ["val", "backtest"]:
        period_df = split_data[period_name]
        # Clientes que también tienen historial en train
        train_hist = build_customer_history(split_data["train"])
        period_customers = set(period_df["customer_unique_id"].unique())
        period_truth = (period_df.groupby("customer_unique_id")["category_en"]
                        .apply(set).reset_index()
                        .rename(columns={"category_en": "true_cats"}))

        eval_users = train_hist[
            train_hist["customer_unique_id"].isin(period_customers)
        ]

        if len(eval_users) == 0:
            continue

        hits, total = 0, 0
        for _, row in eval_users.iterrows():
            cust = row["customer_unique_id"]
            train_cats = list(set(row["category_history"]))
            recs = set(recommend_svd(train_cats, cat_sim_df, top_k=10))
            truth_row = period_truth[period_truth["customer_unique_id"] == cust]
            if len(truth_row) > 0:
                truth = truth_row.iloc[0]["true_cats"]
                hits  += len(recs & truth)
                total += len(truth)

        results.append({
            "Periodo":    period_name.capitalize(),
            "Usuarios":   len(eval_users),
            "Recall@10":  round(hits / total, 4) if total > 0 else 0.0,
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# 7. SIMULACIÓN MENSUAL (nuevo lote de datos)
# ─────────────────────────────────────────────

def simulate_monthly_update(master: pd.DataFrame,
                             n_components: int = 15) -> pd.DataFrame:
    """
    Simula el reentrenamiento mensual del modelo:
    - Acumula datos mes a mes desde el mes 12 en adelante
    - Registra Recall@10 del modelo reentrenado en cada punto
    """
    months = sorted(master.dropna(subset=["order_ym"])["order_ym"].unique())
    results = []

    # Solo simulamos desde que tenemos datos suficientes
    for i in range(12, len(months)):
        cutoff  = months[i]
        train   = master[master["order_ym"] <= cutoff]
        hist    = build_customer_history(train)
        multi   = hist[hist["n_categories"] >= 2]
        if len(multi) < 30:
            continue

        matrix, _, cats, _, _ = build_interaction_matrix(train)
        _, cat_factors, _     = train_svd_model(matrix, n_components=n_components)
        cat_sim               = compute_category_similarity(cat_factors, cats)
        _, pop                = compute_popularity_baseline(train, top_k=20)
        m                    = evaluate_svd_model(multi, cat_sim, pop, k=10)
        m_base               = precision_recall_at_k(multi, pop, k=10)

        results.append({
            "Mes":              cutoff,
            "Clientes_train":   train["customer_unique_id"].nunique(),
            "Pedidos_train":    train["order_id"].nunique(),
            "Usuarios_eval":    m["n_users"],
            "Recall_SVD":       m["Recall@10"],
            "Recall_Baseline":  m_base["Recall@10"],
            "Precision_SVD":    m["Precision@10"],
            "Precision_Baseline": m_base["Precision@10"],
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# 8. PIPELINE COMPLETO REPRODUCIBLE
# ─────────────────────────────────────────────

def run_full_pipeline(n_components: int = 15) -> dict:
    """
    Ejecuta el pipeline completo de Sprint 2.
    Retorna todos los artefactos necesarios para el dashboard.
    """
    print("[1/7] Cargando datos...")
    orders, customers, items, products, reviews, payments, sellers, cat_map = load_raw_tables()
    master_raw = build_master_table(orders, customers, items, products,
                                    reviews, payments, sellers, cat_map)

    print("[2/7] Limpieza de datos...")
    cleaning_report, master = get_cleaning_report(master_raw)

    print("[3/7] Feature engineering (RFM)...")
    rfm        = build_rfm_features(master)
    cat_feat   = build_category_features(master)

    print("[4/7] Construyendo matriz de interacciones...")
    matrix, users, cats, u2i, c2i = build_interaction_matrix(master)
    stats = matrix_stats(matrix, users, cats)

    print("[5/7] Entrenando modelo SVD...")
    svd, cat_factors, user_factors = train_svd_model(matrix, n_components=n_components)
    cat_sim = compute_category_similarity(cat_factors, cats)

    print("[6/7] Evaluando modelos...")
    hist     = build_customer_history(master)
    _, pop   = compute_popularity_baseline(master, top_k=20)
    cmp      = compare_models(hist, cat_sim, pop)
    splits   = build_temporal_splits(master)
    split_eval = evaluate_on_split(splits, cat_sim, pop)

    print("[7/7] Simulacion mensual...")
    monthly = simulate_monthly_update(master_raw, n_components=n_components)

    print("Pipeline completado.")
    return {
        "master_raw":      master_raw,
        "master_clean":    master,
        "cleaning_report": cleaning_report,
        "rfm":             rfm,
        "cat_feat":        cat_feat,
        "matrix":          matrix,
        "users":           users,
        "cats":            cats,
        "matrix_stats":    stats,
        "svd":             svd,
        "cat_factors":     cat_factors,
        "user_factors":    user_factors,
        "cat_sim":         cat_sim,
        "customer_history": hist,
        "popular_fallback": pop,
        "model_comparison": cmp,
        "splits":          splits,
        "split_eval":      split_eval,
        "monthly_sim":     monthly,
        "n_components":    n_components,
    }


if __name__ == "__main__":
    artifacts = run_full_pipeline(n_components=15)
    print("\nComparacion de modelos:")
    print(artifacts["model_comparison"].to_string(index=False))
    print("\nEvaluacion por periodo:")
    print(artifacts["split_eval"].to_string(index=False))
