"""
Dashboard Sprint 1 – Recomendación de Productos Personalizados
Olist Brazilian E-Commerce | Maestría Data Science
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sprint1_eda import (
    load_raw_tables,
    build_master_table,
    build_user_item_table,
    build_customer_history,
    compute_popularity_baseline,
    precision_recall_at_k,
    build_copurchase_matrix,
    data_quality_report,
    business_metrics,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sprint 1 · Recomendación de Productos · Olist",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = px.colors.qualitative.Bold
BG_CARD = "#F0F2F6"

# ─────────────────────────────────────────────
# CARGA CON CACHE
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando datos Olist…")
def get_data():
    orders, customers, items, products, reviews, payments, sellers, category_map = load_raw_tables()
    master = build_master_table(orders, customers, items, products, reviews,
                                payments, sellers, category_map)
    return master

@st.cache_data(show_spinner="Calculando métricas baseline…")
def get_baseline(master_hash):
    _ = master_hash  # trigger cache key
    master = get_data()
    top10, full_pop = compute_popularity_baseline(master, top_k=20)
    hist             = build_customer_history(master)
    copurchase       = build_copurchase_matrix(master, min_support=30)
    metrics_k10      = precision_recall_at_k(hist, full_pop, k=10)
    metrics_k5       = precision_recall_at_k(hist, full_pop, k=5)
    metrics_k20      = precision_recall_at_k(hist, full_pop, k=20)
    ui_table         = build_user_item_table(master)
    dq               = data_quality_report(master)
    bm               = business_metrics(master)
    return top10, full_pop, hist, copurchase, metrics_k10, metrics_k5, metrics_k20, ui_table, dq, bm

master = get_data()
(top10, full_pop, hist, copurchase,
 metrics_k10, metrics_k5, metrics_k20,
 ui_table, dq, bm) = get_baseline(len(master))

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("Sprint 1")
    st.caption("Caso #8 · Recomendación de Productos Personalizados")
    st.divider()
    st.markdown("**Dataset:** Olist Brazilian E-Commerce")
    st.markdown(f"**Registros Master Table:** {len(master):,}")
    st.markdown(f"**Clientes únicos:** {master['customer_unique_id'].nunique():,}")
    st.markdown(f"**Productos:** {master['product_id'].nunique():,}")
    st.markdown(f"**Categorías:** {master['category_en'].nunique()}")
    st.divider()
    st.markdown("Integrantes")
    st.markdown("Gerson Jesus Torrez Marca")
    st.markdown("Juan Marcos Miranda Nina")

# ─────────────────────────────────────────────
# TABS PRINCIPALES
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Definición del Problema",
    "EDA & Master Table",
    "Análisis de Productos",
    "Análisis de Clientes",
    "Baseline del Modelo",
    "Métricas de Negocio",
])


# ══════════════════════════════════════════════
# TAB 1: DEFINICIÓN DEL PROBLEMA
# ══════════════════════════════════════════════
with tab1:
    st.header("Sprint 1 – Definición del Problema de Negocio")

    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.subheader("Problema de Negocio")
        st.info("""
**¿Cómo podemos sugerir productos relevantes a cada cliente de Olist,
aumentando la probabilidad de recompra y el ticket promedio?**

Olist presenta una tasa de recompra muy baja (~3%). La mayoría de los clientes
realizan una sola compra. Un sistema de recomendación personalizado puede:
- Aumentar la tasa de recompra mediante sugerencias relevantes post-compra.
- Incrementar el cross-selling entre categorías complementarias.
- Mejorar la experiencia del cliente y la satisfacción (review_score).
        """)

        st.subheader("Objetivo del MVP")
        st.success("""
Construir un sistema de recomendación de **categorías de productos** que,
dado el historial de compras de un cliente, sugiera las **top-K categorías**
con mayor probabilidad de interés, con temporalidad **continua** (se actualiza
al registrarse nuevas compras).

**Target:** `category_en` (categoría recomendada siguiente)
**Tipo:** Sistema de recomendación híbrido (popularidad + filtrado colaborativo)
        """)

    with col_b:
        st.subheader("Hipótesis")
        hypotheses = [
            ("H1", "Clientes que compran en categorías de hogar (bed_bath_table, furniture) "
             "también compran en categorías de decoración con alta frecuencia."),
            ("H2", "Un mayor número de ítems comprados previamente implica una mayor "
             "diversidad de categorías exploradas."),
            ("H3", "Clientes con alta satisfacción (review_score ≥ 4) tienen mayor "
             "probabilidad de recompra y de explorar nuevas categorías."),
            ("H4", "Existe un patrón estacional en las categorías más compradas "),
            ("H5", "Las categorías de electrónica y accesorios muestran alta "
             "co-compra entre sí."),
        ]
        for code, text in hypotheses:
            st.markdown(f"**{code}** — {text}")

    st.divider()
    st.subheader("Variables Clave – Master Table (60 metadatos)")

    feature_groups = {
        "Cliente": ["customer_unique_id", "customer_state", "customer_city",
                    "customer_zip_code_prefix"],
        "Pedido":  ["order_id", "order_status", "order_purchase_timestamp",
                    "order_year", "order_month", "order_ym"],
        "Producto": ["product_id", "category_en", "product_category_name",
                     "product_weight_g", "product_photos_qty"],
        "Transacción": ["price", "freight_value", "item_total", "freight_ratio",
                        "total_payment", "num_installments", "payment_type"],
        "Reseña": ["review_score", "review_count"],
        "Logística": ["delivery_days", "estimated_days", "delivery_delay", "is_late",
                      "seller_state"],
    }

    cols = st.columns(3)
    for i, (group, features) in enumerate(feature_groups.items()):
        with cols[i % 3]:
            st.markdown(f"**{group}**")
            for f in features:
                st.markdown(f"  - `{f}`")

    st.divider()
    st.subheader("Pipeline de Automatización Mensual")
    st.code("""
┌──────────────────────────────────────────────────────────┐
│  PIPELINE MENSUAL – Recomendación de Productos Olist      │
├──────────────────────────────────────────────────────────┤
│  1. Ingesta nuevos CSVs  ──►  Validación calidad          │
│  2. ETL / Feature Engineering  ──►  Master Table          │
│  3. Actualizar matriz usuario-categoría                   │
│  4. Re-entrenar / actualizar modelo de recomendación      │
│  5. Generar top-K recomendaciones por segmento            │
│  6. Exportar resultados  ──►  API / Dashboard             │
│  7. Registrar métricas en MLflow                          │
└──────────────────────────────────────────────────────────┘
    """, language="text")


# ══════════════════════════════════════════════
# TAB 2: EDA & MASTER TABLE
# ══════════════════════════════════════════════
with tab2:
    st.header("EDA & Calidad de la Master Table")

    # KPIs rápidos
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Pedidos totales",      f"{bm['total_orders']:,}")
    k2.metric("Clientes únicos",      f"{bm['total_customers']:,}")
    k3.metric("Productos",            f"{bm['total_products']:,}")
    k4.metric("Categorías",           f"{bm['total_categories']}")
    k5.metric("Ticket promedio (BRL)",f"{bm['avg_order_value_brl']}")
    k6.metric("Satisfacción media",   f"{bm['avg_review_score']} / 5")

    st.divider()

    # Calidad de datos
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Calidad de Datos – Valores Nulos")
        dq_plot = dq[dq["null_pct"] > 0].sort_values("null_pct", ascending=True)
        if len(dq_plot) > 0:
            fig_dq = px.bar(dq_plot, x="null_pct", y="column", orientation="h",
                            color="null_pct", color_continuous_scale="Reds",
                            labels={"null_pct": "% Nulos", "column": "Variable"},
                            title="Variables con valores nulos (%)")
            fig_dq.update_layout(height=450, showlegend=False,
                                  coloraxis_showscale=False)
            st.plotly_chart(fig_dq, use_container_width=True)
        else:
            st.success("No hay valores nulos en la Master Table.")

    with col2:
        st.subheader("Resumen de calidad")
        st.dataframe(dq[["column", "null_pct", "unique"]].head(20),
                     use_container_width=True, height=450)

    st.divider()

    # Volumen temporal
    st.subheader("Volumen de Pedidos en el Tiempo")
    orders_ts = (master.groupby("order_ym")["order_id"]
                 .nunique().reset_index()
                 .rename(columns={"order_id": "orders"}))
    orders_ts = orders_ts[orders_ts["order_ym"] != "NaT"]

    fig_ts = px.area(orders_ts, x="order_ym", y="orders",
                     title="Pedidos mensuales (2016-2018)",
                     labels={"order_ym": "Mes", "orders": "Pedidos"},
                     color_discrete_sequence=[PALETTE[0]])
    fig_ts.update_xaxes(tickangle=45)
    fig_ts.update_layout(height=320)
    st.plotly_chart(fig_ts, use_container_width=True)

    st.divider()

    # Distribución de review_score
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Distribución de Review Score")
        rev_dist = (master.dropna(subset=["review_score"])
                    .groupby("review_score")["order_id"].nunique()
                    .reset_index())
        fig_rev = px.bar(rev_dist, x="review_score", y="order_id",
                         labels={"review_score": "Puntuación", "order_id": "Pedidos"},
                         color="review_score",
                         color_continuous_scale="RdYlGn",
                         title="Distribución de calificaciones")
        fig_rev.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig_rev, use_container_width=True)

    with col4:
        st.subheader("Distribución de Días de Entrega")
        del_clean = master.dropna(subset=["delivery_days"])
        del_clean = del_clean[(del_clean["delivery_days"] > 0) &
                              (del_clean["delivery_days"] < 100)]
        fig_del = px.histogram(del_clean, x="delivery_days", nbins=50,
                               title="Días hasta entrega al cliente",
                               labels={"delivery_days": "Días", "count": "Pedidos"},
                               color_discrete_sequence=[PALETTE[2]])
        fig_del.add_vline(x=del_clean["delivery_days"].median(),
                          line_dash="dash", line_color="red",
                          annotation_text=f"Mediana: {del_clean['delivery_days'].median():.0f}d")
        fig_del.update_layout(height=320)
        st.plotly_chart(fig_del, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3: ANÁLISIS DE PRODUCTOS
# ══════════════════════════════════════════════
with tab3:
    st.header("Análisis de Productos y Categorías")

    # Top categorías
    cat_orders = (master.groupby("category_en")["order_id"]
                  .nunique().sort_values(ascending=False).reset_index()
                  .rename(columns={"order_id": "num_orders"}))
    cat_orders["pct"] = cat_orders["num_orders"] / cat_orders["num_orders"].sum() * 100

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Top 20 Categorías por Volumen de Pedidos")
        fig_cat = px.bar(cat_orders.head(20), x="num_orders", y="category_en",
                         orientation="h",
                         color="num_orders", color_continuous_scale="Blues",
                         labels={"num_orders": "Pedidos", "category_en": "Categoría"},
                         title="Las 20 categorías más vendidas")
        fig_cat.update_yaxes(categoryorder="total ascending")
        fig_cat.update_layout(height=550, showlegend=False,
                               coloraxis_showscale=False)
        st.plotly_chart(fig_cat, use_container_width=True)

    with col2:
        st.subheader("Treemap de Categorías")
        fig_tree = px.treemap(cat_orders.head(30), path=["category_en"],
                              values="num_orders",
                              color="pct", color_continuous_scale="RdBu",
                              title="Distribución de ventas por categoría")
        fig_tree.update_layout(height=550)
        st.plotly_chart(fig_tree, use_container_width=True)

    st.divider()

    # Ticket promedio por categoría
    st.subheader("Ticket Promedio (BRL) por Categoría – Top 20")
    cat_price = (master.groupby("category_en")["item_total"]
                 .mean().sort_values(ascending=False).reset_index()
                 .rename(columns={"item_total": "avg_ticket"}))
    cat_price["avg_ticket"] = cat_price["avg_ticket"].round(2)

    fig_price = px.bar(cat_price.head(20), x="category_en", y="avg_ticket",
                       color="avg_ticket", color_continuous_scale="Greens",
                       labels={"category_en": "Categoría", "avg_ticket": "Precio medio (BRL)"},
                       title="Ticket promedio por categoría (top 20)")
    fig_price.update_xaxes(tickangle=45)
    fig_price.update_layout(height=380, coloraxis_showscale=False)
    st.plotly_chart(fig_price, use_container_width=True)

    st.divider()

    # Mapa de calor: categoría × mes
    st.subheader("Demanda por Categoría y Mes (Heatmap)")
    top_cats_list = cat_orders.head(15)["category_en"].tolist()
    heat_df = (master[master["category_en"].isin(top_cats_list)]
               .groupby(["order_month", "category_en"])["order_id"]
               .nunique().reset_index()
               .rename(columns={"order_id": "orders"}))
    heat_pivot = heat_df.pivot(index="category_en", columns="order_month",
                               values="orders").fillna(0)
    month_names = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
                   7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}
    heat_pivot.columns = [month_names.get(c, c) for c in heat_pivot.columns]

    fig_heat = px.imshow(heat_pivot,
                         color_continuous_scale="YlOrRd",
                         title="Pedidos por categoría y mes",
                         labels={"x": "Mes", "y": "Categoría", "color": "Pedidos"},
                         aspect="auto")
    fig_heat.update_layout(height=480)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    # Matriz de co-compra
    st.subheader("Co-Compra entre Categorías (Top 25 pares)")
    st.caption("Número de clientes que compraron ambas categorías en el mismo historial.")
    copurchase_top = copurchase.head(25)
    fig_cop = px.bar(copurchase_top, x="support", y=copurchase_top["cat_a"] + " ↔ " + copurchase_top["cat_b"],
                     orientation="h",
                     color="support", color_continuous_scale="Purples",
                     labels={"support": "Clientes", "y": "Par de categorías"},
                     title="Top 25 pares de categorías co-compradas")
    fig_cop.update_yaxes(categoryorder="total ascending")
    fig_cop.update_layout(height=560, coloraxis_showscale=False)
    st.plotly_chart(fig_cop, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 4: ANÁLISIS DE CLIENTES
# ══════════════════════════════════════════════
with tab4:
    st.header("Análisis de Clientes")

    col1, col2, col3 = st.columns(3)
    col1.metric("Clientes únicos", f"{master['customer_unique_id'].nunique():,}")
    col2.metric("Tasa de recompra", f"{bm['repeat_customer_rate']} %")
    col3.metric("Categ. promedio / cliente",
                f"{bm['avg_categories_per_customer']:.2f}")

    st.divider()

    # Distribución de compras por cliente
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Distribución de Pedidos por Cliente")
        orders_per_cust = (master.groupby("customer_unique_id")["order_id"]
                           .nunique().value_counts().reset_index()
                           .rename(columns={"order_id": "n_orders",
                                            "count": "n_customers"}))
        orders_per_cust = orders_per_cust[orders_per_cust["n_orders"] <= 10]
        fig_opc = px.bar(orders_per_cust.sort_values("n_orders"),
                         x="n_orders", y="n_customers",
                         labels={"n_orders": "Pedidos realizados",
                                 "n_customers": "Número de clientes"},
                         color="n_orders", color_continuous_scale="Teal",
                         title="Clientes según número de pedidos")
        fig_opc.update_layout(height=350, showlegend=False,
                               coloraxis_showscale=False)
        st.plotly_chart(fig_opc, use_container_width=True)

    with col_b:
        st.subheader("Diversidad de Categorías por Cliente")
        cats_per_cust = (master.groupby("customer_unique_id")["category_en"]
                         .nunique().value_counts().reset_index()
                         .rename(columns={"category_en": "n_cats",
                                          "count": "n_customers"}))
        cats_per_cust = cats_per_cust[cats_per_cust["n_cats"] <= 10]
        fig_cpc = px.bar(cats_per_cust.sort_values("n_cats"),
                         x="n_cats", y="n_customers",
                         labels={"n_cats": "Categorías distintas",
                                 "n_customers": "Número de clientes"},
                         color="n_cats", color_continuous_scale="Magenta",
                         title="Clientes según diversidad de categorías")
        fig_cpc.update_layout(height=350, showlegend=False,
                               coloraxis_showscale=False)
        st.plotly_chart(fig_cpc, use_container_width=True)

    st.divider()

    # Mapa por estado
    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Clientes por Estado Brasileño")
        cust_state = (master.groupby("customer_state")["customer_unique_id"]
                      .nunique().sort_values(ascending=False).reset_index()
                      .rename(columns={"customer_unique_id": "clientes"}))
        fig_state = px.bar(cust_state, x="customer_state", y="clientes",
                           color="clientes", color_continuous_scale="Blues",
                           labels={"customer_state": "Estado", "clientes": "Clientes"},
                           title="Distribución geográfica de clientes")
        fig_state.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_state, use_container_width=True)

    with col_d:
        st.subheader("Gasto Total por Estado (BRL)")
        spend_state = (master.groupby("customer_state")["item_total"]
                       .sum().sort_values(ascending=False).reset_index()
                       .rename(columns={"item_total": "total_brl"}))
        spend_state["total_brl"] = spend_state["total_brl"].round(0)
        fig_spend = px.bar(spend_state, x="customer_state", y="total_brl",
                           color="total_brl", color_continuous_scale="Greens",
                           labels={"customer_state": "Estado",
                                   "total_brl": "Gasto total (BRL)"},
                           title="Gasto total por estado")
        fig_spend.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_spend, use_container_width=True)

    st.divider()

    # Segmentación por historial
    st.subheader("Perfil de Clientes para Recomendación")
    hist_seg = hist.copy()
    hist_seg["unique_cats"] = hist_seg["category_history"].apply(lambda x: len(set(x)))
    hist_seg["segment"] = pd.cut(hist_seg["unique_cats"],
                                  bins=[0, 1, 2, 5, 100],
                                  labels=["1 categoría", "2 categorías",
                                          "3-5 categorías", "6+ categorías"])
    seg_counts = hist_seg["segment"].value_counts().reset_index()
    fig_seg = px.pie(seg_counts, names="segment", values="count",
                     title="Segmentos de clientes según diversidad de compras",
                     color_discrete_sequence=PALETTE,
                     hole=0.4)
    fig_seg.update_layout(height=380)
    st.plotly_chart(fig_seg, use_container_width=True)

    st.info("""
**Insight para el sistema de recomendación:**
La mayoría de los clientes (~97%) tiene solo 1 pedido (1 categoría).
Esto favorece un enfoque **híbrido**: popularidad global para nuevos clientes
y filtrado colaborativo para los recurrentes.
    """)


# ══════════════════════════════════════════════
# TAB 5: BASELINE DEL MODELO
# ══════════════════════════════════════════════
with tab5:
    st.header("Baseline del Sistema de Recomendación")

    st.subheader("Estrategia Baseline: Popularidad Global")
    st.markdown("""
    La recomendación baseline propone las **K categorías más compradas** en el histórico.
    Es el punto de partida para comparar modelos más sofisticados (CF, content-based, híbrido).
    """)

    # Métricas por K
    col1, col2, col3 = st.columns(3)
    col1.metric("Precision@5",  f"{metrics_k5[f'Precision@5']:.4f}",
                help="Proporción de recomendaciones que son relevantes (k=5)")
    col2.metric("Precision@10", f"{metrics_k10[f'Precision@10']:.4f}",
                help="Proporción de recomendaciones que son relevantes (k=10)")
    col3.metric("Precision@20", f"{metrics_k20[f'Precision@20']:.4f}",
                help="Proporción de recomendaciones que son relevantes (k=20)")

    col4, col5, col6 = st.columns(3)
    col4.metric("Recall@5",  f"{metrics_k5[f'Recall@5']:.4f}",
                help="Proporción de ítems relevantes recuperados (k=5)")
    col5.metric("Recall@10", f"{metrics_k10[f'Recall@10']:.4f}",
                help="Proporción de ítems relevantes recuperados (k=10)")
    col6.metric("Recall@20", f"{metrics_k20[f'Recall@20']:.4f}",
                help="Proporción de ítems relevantes recuperados (k=20)")

    st.caption(f"Evaluado con {metrics_k10['n_users_evaluated']:,} clientes "
               f"(leave-one-out sobre clientes con ≥2 categorías distintas)")

    st.divider()

    # Curva Precision / Recall vs K
    st.subheader("Curva Precision & Recall vs K")
    k_values = list(range(1, 21))
    prec_vals, rec_vals = [], []
    for k in k_values:
        m = precision_recall_at_k(hist, full_pop, k=k)
        prec_vals.append(m[f"Precision@{k}"])
        rec_vals.append(m[f"Recall@{k}"])

    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=k_values, y=prec_vals, name="Precision@K",
                                 mode="lines+markers",
                                 line=dict(color=PALETTE[0], width=2)))
    fig_pr.add_trace(go.Scatter(x=k_values, y=rec_vals, name="Recall@K",
                                 mode="lines+markers",
                                 line=dict(color=PALETTE[1], width=2)))
    fig_pr.update_layout(
        title="Precision y Recall del baseline según K recomendaciones",
        xaxis_title="K (número de recomendaciones)",
        yaxis_title="Valor",
        height=380,
        legend=dict(x=0.7, y=0.5),
    )
    st.plotly_chart(fig_pr, use_container_width=True)

    st.divider()

    # Top-20 categorías recomendadas
    st.subheader("Top 20 Categorías – Recomendación Baseline")
    col_t, col_bar = st.columns([1, 2])
    with col_t:
        st.dataframe(
            top10[["rank", "category_en", "num_orders", "pct_of_total"]]
            .rename(columns={"rank": "Rank", "category_en": "Categoría",
                              "num_orders": "Pedidos", "pct_of_total": "% Total"}),
            use_container_width=True, hide_index=True, height=480
        )
    with col_bar:
        fig_top = px.funnel(top10, x="num_orders", y="category_en",
                            color_discrete_sequence=[PALETTE[3]],
                            labels={"num_orders": "Pedidos", "category_en": "Categoría"},
                            title="Embudo de popularidad – top 20 categorías")
        fig_top.update_layout(height=480)
        st.plotly_chart(fig_top, use_container_width=True)

    st.divider()

    # Roadmap de mejora
    st.subheader("Roadmap del Modelo – Evolución Sprint a Sprint")
    roadmap = pd.DataFrame({
        "Sprint": ["Sprint 1", "Sprint 2", "Sprint 3", "Sprint 4"],
        "Modelo": [
            "Baseline popularidad global",
            "Filtrado colaborativo usuario-ítem (SVD/ALS)",
            "Modelo híbrido (CF + Content-based)",
            "API de recomendación en tiempo real",
        ],
        "Precision@10 esperado": ["0.063", "0.10–0.15", "0.15–0.22", "≥0.20 + A/B test"],
        "Estado": ["Completado", "Próximo", "Futuro", "Futuro"],
    })
    st.dataframe(roadmap, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 6: MÉTRICAS DE NEGOCIO
# ══════════════════════════════════════════════
with tab6:
    st.header("Métricas de Negocio – Baseline")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("KPIs del Negocio")
        bm_df = pd.DataFrame([
            {"Métrica": "Total de pedidos",          "Valor": f"{bm['total_orders']:,}"},
            {"Métrica": "Clientes únicos",            "Valor": f"{bm['total_customers']:,}"},
            {"Métrica": "Productos catálogo",         "Valor": f"{bm['total_products']:,}"},
            {"Métrica": "Categorías activas",         "Valor": str(bm["total_categories"])},
            {"Métrica": "Ticket promedio (BRL)",      "Valor": str(bm["avg_order_value_brl"])},
            {"Métrica": "Ítems promedio / pedido",    "Valor": str(bm["avg_items_per_order"])},
            {"Métrica": "Satisfacción media",         "Valor": f"{bm['avg_review_score']} / 5.0"},
            {"Métrica": "Entrega promedio (días)",    "Valor": str(bm["avg_delivery_days"])},
            {"Métrica": "Tasa de retraso (%)",        "Valor": f"{bm['late_delivery_rate_pct']} %"},
            {"Métrica": "Tasa de recompra (%)",       "Valor": f"{bm['repeat_customer_rate']} %"},
            {"Métrica": "Estado principal",           "Valor": bm["top_state_customers"]},
        ])
        st.dataframe(bm_df, use_container_width=True, hide_index=True, height=440)

    with col2:
        st.subheader("Impacto Esperado del Sistema de Recomendación")
        st.success("""
**Oportunidades de negocio identificadas:**

1. **Aumento de recompra (+2-5%)**
   - La tasa actual es ~3%. Con recomendaciones post-compra relevantes,
     se puede incrementar a 5-8%.

2. **Cross-sell entre categorías complementarias**
   - Categorías con alta co-compra (bed/bath ↔ furniture, electronics ↔ accessories)
     son candidatas prioritarias para recomendación cruzada.

3. **Reducción de churn silencioso**
   - Notificaciones personalizadas basadas en patrones de compra previos.

4. **Incremento de ticket promedio**
   - Si 5% de los 96K clientes hace una segunda compra de BRL 160:
     **+BRL 768.000 mensuales estimados**.

5. **Segmentación para campañas de marketing**
   - Clientes con ≥2 categorías son el público objetivo principal.
        """)

    st.divider()

    # Análisis de ingresos mensuales
    st.subheader("Ingresos Mensuales por Categoría – Top 10")
    revenue_monthly = (master.groupby(["order_ym", "category_en"])["item_total"]
                       .sum().reset_index()
                       .rename(columns={"item_total": "revenue"}))
    top_cats_rev = (master.groupby("category_en")["item_total"]
                    .sum().nlargest(10).index.tolist())
    rev_filtered = revenue_monthly[revenue_monthly["category_en"].isin(top_cats_rev)]
    rev_filtered = rev_filtered[rev_filtered["order_ym"] != "NaT"]

    fig_rev_ts = px.line(rev_filtered, x="order_ym", y="revenue",
                         color="category_en", markers=True,
                         labels={"order_ym": "Mes", "revenue": "Ingresos (BRL)",
                                 "category_en": "Categoría"},
                         title="Ingresos mensuales – top 10 categorías",
                         color_discrete_sequence=PALETTE)
    fig_rev_ts.update_xaxes(tickangle=45)
    fig_rev_ts.update_layout(height=420, legend=dict(font_size=10))
    st.plotly_chart(fig_rev_ts, use_container_width=True)

    st.divider()

    # Conclusiones Sprint 1
    st.subheader("Conclusiones Sprint 1")
    st.info("""
**Hallazgos principales del EDA:**

- El dataset cubre **99.441 pedidos** de **96.096 clientes únicos** entre 2016-2018.
- La **tasa de recompra es extremadamente baja (3.1%)**, lo que valida la necesidad
  de un sistema de recomendación para fomentar la reactivación.
- Las categorías **bed_bath_table, health_beauty y sports_leisure** dominan el volumen.
- **Alta concentración geográfica** en SP (São Paulo), lo que permite personalizar
  recomendaciones por región.
- El análisis de co-compra revela pares naturales de categorías complementarias
  que pueden usarse como señales en el modelo.
- **Precision@10 baseline = 0.063** – hay margen significativo de mejora en
  Sprint 2-3 con filtrado colaborativo.

**Target confirmado:** `category_en` (categoría a recomendar como siguiente compra).
**Temporalidad:** Continua (actualización en tiempo real o batch diario).
    """)
