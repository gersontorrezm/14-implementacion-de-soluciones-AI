"""
Dashboard Sprint 2 – Feature Engineering & Pipeline Reproducible
Caso #8: Recomendación de Productos Personalizados
Olist Brazilian E-Commerce | Maestría Data Science
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from sprint2_pipeline import (
    run_full_pipeline,
    recommend_svd,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sprint 2 · Pipeline · Recomendación Olist",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE   = px.colors.qualitative.Bold
PALETTE_D = px.colors.qualitative.Dark2

# ─────────────────────────────────────────────
# CACHE – ejecuta pipeline una sola vez
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Ejecutando pipeline Sprint 2 (puede tomar ~2 min)...")
def load_artifacts():
    return run_full_pipeline(n_components=15)

art = load_artifacts()

master      = art["master_clean"]
master_raw  = art["master_raw"]
clean_rep   = art["cleaning_report"]
rfm         = art["rfm"]
cat_feat    = art["cat_feat"]
matrix      = art["matrix"]
stats       = art["matrix_stats"]
svd_model   = art["svd"]
cat_factors = art["cat_factors"]
cat_sim     = art["cat_sim"]
hist        = art["customer_history"]
pop         = art["popular_fallback"]
cmp         = art["model_comparison"]
splits      = art["splits"]
split_eval  = art["split_eval"]
monthly     = art["monthly_sim"]
cats        = art["cats"]

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("Sprint 2")
    st.caption("Caso #8 · Recomendación de Productos")
    st.caption("Gerson Jesus Torrez Marca")
    st.divider()
    st.markdown("**Pipeline version:** 2.0")
    st.markdown(f"**Filas limpias:** {len(master):,}")
    st.markdown(f"**Clientes:** {master['customer_unique_id'].nunique():,}")
    st.markdown(f"**Categorías:** {master['category_en'].nunique()}")
    st.markdown(f"**Componentes SVD:** {art['n_components']}")
    st.divider()
    st.markdown("**Division temporal**")
    for _, row in splits["summary"].iterrows():
        st.markdown(f"- {row['Periodo']}: {row['Desde']} → {row['Hasta']}")
    st.divider()
    st.markdown("**Modelo:** SVD item-item (filtrado colaborativo)")
    st.markdown("**Target:** `category_en` (confirmado)")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Pipeline de Datos",
    "Feature Engineering (RFM)",
    "Modelo SVD",
    "Comparacion de Modelos",
    "Simulacion Mensual",
    "Target Final & Metricas",
])


# ══════════════════════════════════════════════
# TAB 1 – PIPELINE DE DATOS
# ══════════════════════════════════════════════
with tab1:
    st.header("Sprint 2 – Pipeline de Ingenieria de Datos")

    st.subheader("Diagrama del Pipeline Reproducible")
    st.code("""
PIPELINE SPRINT 2 – Recomendacion de Productos (ejecucion mensual)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [FUENTE]  CSVs Olist (9 tablas)
      
  [PASO 1] load_raw_tables()           ← carga y parseo de fechas
      
  [PASO 2] build_master_table()        ← join de 8 datasets
           + variables derivadas       ← delivery_days, is_late,
                                          item_total, freight_ratio
      
  [PASO 3] clean_master_table()        ← eliminar cancelados
           - nulos delivery_days       ← imputar mediana por estado
           - nulos review_score        ← imputar mediana global
           - outliers precio           ← cap IQR×3
      
  [PASO 4] build_rfm_features()        ← Recency / Frequency / Monetary
           build_category_features()   ← popularidad, ticket, review
      
  [PASO 5] build_interaction_matrix()  ← matriz dispersa user × category
      
  [PASO 6] train_svd_model()           ← TruncatedSVD 15 componentes
           compute_category_similarity()  ← cosine similarity embeddings
      
  [PASO 7] evaluate & export           ← Precision@K, Recall@K
                                          artefactos pkl, metricas MLflow

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """, language="text")

    st.divider()

    # Informe de limpieza
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Informe de Limpieza – Antes vs Despues")
        st.dataframe(clean_rep, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Decisiones de Limpieza")
        st.info("""
**Filas eliminadas:**
- Pedidos sin categoria: no aportan al sistema de recomendacion
- Estados cancelados / unavailable: no representan compras reales

**Imputaciones:**
- `delivery_days`: mediana por estado del vendedor (proxy geografico)
- `review_score`: mediana global (4.0) para pedidos sin resena

**Outliers de precio:**
- Limite = Q3 + 3 x IQR
- Elimina productos con precios atipicamente altos que distorsionan el ticket promedio

**Resultado:** reduccion ~5% de filas, 0% de nulos en variables clave
        """)

    st.divider()

    # Comparación nulos antes/después
    st.subheader("Variables con Nulos – Antes vs Despues de Limpieza")
    vars_key = ["delivery_days", "review_score", "delivery_delay",
                "estimated_days", "freight_ratio"]
    null_before = [master_raw[v].isnull().mean() * 100 for v in vars_key]
    null_after  = [master[v].isnull().mean() * 100 for v in vars_key]

    fig_null = go.Figure()
    fig_null.add_trace(go.Bar(name="Antes", x=vars_key, y=null_before,
                               marker_color=PALETTE[3]))
    fig_null.add_trace(go.Bar(name="Despues", x=vars_key, y=null_after,
                               marker_color=PALETTE[0]))
    fig_null.update_layout(
        barmode="group", height=350,
        title="Porcentaje de nulos antes y despues de limpieza",
        yaxis_title="% Nulos",
    )
    st.plotly_chart(fig_null, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 – FEATURE ENGINEERING (RFM)
# ══════════════════════════════════════════════
with tab2:
    st.header("Feature Engineering – Analisis RFM")

    st.subheader("Variables Creadas en Sprint 2")
    feat_table = pd.DataFrame({
        "Variable": [
            "recency", "frequency", "monetary", "avg_ticket",
            "avg_review", "category_diversity", "total_items",
            "avg_delivery_days", "late_rate", "preferred_category",
            "R_score", "F_score", "M_score", "RFM_score", "segment",
        ],
        "Descripcion": [
            "Dias desde la ultima compra",
            "Numero total de pedidos distintos",
            "Gasto total acumulado (BRL)",
            "Gasto promedio por pedido",
            "Review promedio dado por el cliente",
            "Numero de categorias distintas compradas",
            "Numero total de items comprados",
            "Dias de entrega promedio experimentados",
            "Proporcion de pedidos con retraso",
            "Categoria con mas compras del cliente",
            "Puntaje de Recency (1=mas antiguo, 5=mas reciente)",
            "Puntaje de Frequency (1=menos compras, 5=mas)",
            "Puntaje de Monetary (1=menor gasto, 5=mayor)",
            "Promedio de R+F+M (escala 1-5)",
            "Segmento: Perdidos / En riesgo / Regulares / Leales / Campeones",
        ],
        "Tipo": [
            "Continua","Discreta","Continua","Continua",
            "Continua","Discreta","Discreta",
            "Continua","Continua","Categorica",
            "Ordinal","Ordinal","Ordinal","Continua","Categorica",
        ],
    })
    st.dataframe(feat_table, use_container_width=True, hide_index=True)

    st.divider()

    # Distribucion RFM
    col1, col2, col3 = st.columns(3)
    with col1:
        fig_r = px.histogram(rfm, x="recency", nbins=50,
                             title="Distribucion de Recency (dias)",
                             color_discrete_sequence=[PALETTE[0]],
                             labels={"recency": "Dias desde ultima compra"})
        fig_r.add_vline(x=rfm["recency"].median(), line_dash="dash",
                         line_color="red",
                         annotation_text=f"Mediana: {rfm['recency'].median():.0f}d")
        fig_r.update_layout(height=300)
        st.plotly_chart(fig_r, use_container_width=True)
    with col2:
        fig_f = px.histogram(rfm[rfm["frequency"] <= 5], x="frequency",
                             title="Distribucion de Frequency (pedidos)",
                             color_discrete_sequence=[PALETTE[1]],
                             labels={"frequency": "Pedidos totales"})
        fig_f.update_layout(height=300)
        st.plotly_chart(fig_f, use_container_width=True)
    with col3:
        fig_m = px.histogram(rfm[rfm["monetary"] <= rfm["monetary"].quantile(0.95)],
                             x="monetary", nbins=50,
                             title="Distribucion de Monetary (BRL)",
                             color_discrete_sequence=[PALETTE[2]],
                             labels={"monetary": "Gasto total (BRL)"})
        fig_m.add_vline(x=rfm["monetary"].median(), line_dash="dash",
                         line_color="red",
                         annotation_text=f"Mediana: {rfm['monetary'].median():.0f}")
        fig_m.update_layout(height=300)
        st.plotly_chart(fig_m, use_container_width=True)

    st.divider()

    # Segmentos RFM
    col4, col5 = st.columns([2, 3])
    with col4:
        st.subheader("Segmentos de Clientes (RFM)")
        seg_counts = rfm["segment"].value_counts().reset_index()
        seg_counts.columns = ["Segmento", "Clientes"]
        seg_counts["Pct"] = (seg_counts["Clientes"] / seg_counts["Clientes"].sum() * 100).round(1)
        st.dataframe(seg_counts, use_container_width=True, hide_index=True)

        st.caption("**Relevancia para recomendacion:**")
        st.markdown("""
- **Campeones / Leales**: prioridad para recomendaciones personalizadas
- **En riesgo / Perdidos**: campanas de reactivacion con top categorias
- **Regulares**: exploracion de nuevas categorias adyacentes
        """)

    with col5:
        st.subheader("Distribucion de Segmentos")
        color_map = {
            "Campeones": "#2ecc71", "Leales": "#3498db",
            "Regulares": "#f39c12", "En riesgo": "#e67e22", "Perdidos": "#e74c3c",
        }
        seg_order = ["Campeones", "Leales", "Regulares", "En riesgo", "Perdidos"]
        seg_plot = seg_counts[seg_counts["Segmento"].isin(seg_order)].copy()
        seg_plot["Segmento"] = pd.Categorical(seg_plot["Segmento"],
                                               categories=seg_order, ordered=True)
        seg_plot = seg_plot.sort_values("Segmento")
        fig_seg = px.bar(seg_plot, x="Segmento", y="Clientes",
                         color="Segmento",
                         color_discrete_map=color_map,
                         title="Clientes por segmento RFM",
                         text="Pct")
        fig_seg.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_seg.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_seg, use_container_width=True)

    st.divider()

    # Features de categorias
    st.subheader("Features de Categorias (para Content-Based)")
    top20_cat = cat_feat.nlargest(20, "total_orders")
    fig_cf = px.scatter(
        top20_cat, x="avg_price", y="avg_review",
        size="total_orders", color="avg_delivery",
        hover_name="category_en",
        color_continuous_scale="RdYlGn_r",
        labels={"avg_price": "Precio promedio (BRL)",
                "avg_review": "Review promedio",
                "avg_delivery": "Entrega (dias)"},
        title="Categorias: precio vs satisfaccion (tamano = volumen, color = dias entrega)",
    )
    fig_cf.update_layout(height=420)
    st.plotly_chart(fig_cf, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 – MODELO SVD
# ══════════════════════════════════════════════
with tab3:
    st.header("Modelo SVD – Filtrado Colaborativo Item-Item")

    st.subheader("Matriz de Interacciones Usuario-Categoria")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Usuarios", f"{stats['n_users']:,}")
    col2.metric("Categorias", f"{stats['n_categories']}")
    col3.metric("Interacciones", f"{stats['n_interactions']:,}")
    col4.metric("Esparsidad", f"{stats['sparsity_pct']:.2f}%")

    st.warning(f"""
**Hallazgo clave de Sprint 2:** La matriz tiene una esparsidad del {stats['sparsity_pct']:.1f}%.
El {100 - stats['density_pct']*100/stats['density_pct']:.0f}% de los clientes tiene exactamente
1 categoria comprada. Esto limita el filtrado colaborativo clasico y justifica
el enfoque hibrido planificado para Sprint 3.
    """)

    st.divider()

    # Varianza explicada por SVD
    st.subheader("Varianza Explicada por los Componentes SVD")
    explained = svd_model.explained_variance_ratio_
    cum_var = np.cumsum(explained) * 100
    n_comp  = len(explained)

    fig_var = go.Figure()
    fig_var.add_trace(go.Bar(
        x=list(range(1, n_comp + 1)),
        y=explained * 100,
        name="Varianza individual",
        marker_color=PALETTE[0],
    ))
    fig_var.add_trace(go.Scatter(
        x=list(range(1, n_comp + 1)),
        y=cum_var,
        name="Varianza acumulada",
        mode="lines+markers",
        line=dict(color=PALETTE[3], width=2),
        yaxis="y2",
    ))
    fig_var.update_layout(
        title=f"Varianza explicada por {n_comp} componentes SVD",
        xaxis_title="Componente",
        yaxis_title="Varianza individual (%)",
        yaxis2=dict(title="Varianza acumulada (%)", overlaying="y",
                    side="right", range=[0, 105]),
        height=380, legend=dict(x=0.6, y=0.5),
    )
    st.plotly_chart(fig_var, use_container_width=True)
    st.caption(f"Los primeros 15 componentes explican el {cum_var[-1]:.1f}% de la varianza total de la matriz de interacciones.")

    st.divider()

    # Embeddings 2D de categorias (PCA sobre cat_factors)
    st.subheader("Espacio Latente de Categorias (PCA 2D sobre embeddings SVD)")
    pca2 = PCA(n_components=2, random_state=42)
    emb2 = pca2.fit_transform(cat_factors)
    emb_df = pd.DataFrame({"x": emb2[:,0], "y": emb2[:,1], "category": cats})
    emb_df = emb_df.merge(
        cat_feat[["category_en", "total_orders"]],
        left_on="category", right_on="category_en", how="left"
    )
    fig_emb = px.scatter(
        emb_df, x="x", y="y",
        text="category", size="total_orders",
        color="total_orders", color_continuous_scale="Viridis",
        title="Embeddings SVD de categorias en 2D (categorias cercanas = patrones de compra similares)",
        labels={"x": "Componente 1", "y": "Componente 2",
                "total_orders": "Pedidos"},
        size_max=30,
    )
    fig_emb.update_traces(textposition="top center", textfont_size=8)
    fig_emb.update_layout(height=560, coloraxis_showscale=False)
    st.plotly_chart(fig_emb, use_container_width=True)

    st.divider()

    # Mapa de calor de similitud entre top categorias
    st.subheader("Mapa de Similitud entre Categorias (Top 20)")
    top20 = cat_feat.nlargest(20, "total_orders")["category_en"].tolist()
    top20 = [c for c in top20 if c in cat_sim.index]
    sim_sub = cat_sim.loc[top20, top20]

    fig_hm = px.imshow(
        sim_sub.round(3),
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        title="Similitud coseno entre las top 20 categorias (espacio latente SVD)",
        labels={"color": "Similitud"},
        aspect="auto",
    )
    fig_hm.update_layout(height=560)
    st.plotly_chart(fig_hm, use_container_width=True)

    st.divider()

    # Simulador de recomendacion
    st.subheader("Simulador de Recomendaciones SVD")
    col_inp, col_out = st.columns([1, 2])
    with col_inp:
        available_cats = sorted(cats.tolist())
        selected = st.multiselect(
            "Selecciona categorias compradas (simula historial de cliente):",
            available_cats,
            default=available_cats[:2],
        )
        k_sim = st.slider("Numero de recomendaciones (K)", 3, 15, 10)

    with col_out:
        if selected:
            recs = recommend_svd(selected, cat_sim, top_k=k_sim)
            if recs:
                rec_df = pd.DataFrame({
                    "Rank": range(1, len(recs)+1),
                    "Categoria Recomendada": recs,
                    "Score Similitud": [
                        round(cat_sim.loc[c, selected].mean(), 4)
                        if c in cat_sim.index and all(s in cat_sim.columns for s in selected)
                        else 0
                        for c in recs
                    ],
                })
                st.dataframe(rec_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No hay recomendaciones disponibles para esa seleccion.")
        else:
            st.info("Selecciona al menos una categoria para ver recomendaciones.")


# ══════════════════════════════════════════════
# TAB 4 – COMPARACION DE MODELOS
# ══════════════════════════════════════════════
with tab4:
    st.header("Comparacion de Modelos – Baseline vs SVD")

    st.subheader("Tabla de Metricas")
    display_cmp = cmp.rename(columns={
        "K": "K",
        "Precision_baseline": "Precision Baseline",
        "Precision_SVD":      "Precision SVD",
        "Recall_baseline":    "Recall Baseline",
        "Recall_SVD":         "Recall SVD",
        "Delta_Precision":    "Delta Precision",
        "Delta_Recall":       "Delta Recall",
    })
    st.dataframe(display_cmp, use_container_width=True, hide_index=True)

    st.divider()

    # Graficos de comparacion
    col1, col2 = st.columns(2)
    with col1:
        fig_prec = go.Figure()
        fig_prec.add_trace(go.Bar(name="Baseline", x=cmp["K"].astype(str),
                                   y=cmp["Precision_baseline"],
                                   marker_color=PALETTE[3]))
        fig_prec.add_trace(go.Bar(name="SVD", x=cmp["K"].astype(str),
                                   y=cmp["Precision_SVD"],
                                   marker_color=PALETTE[0]))
        fig_prec.update_layout(
            barmode="group",
            title="Precision@K: Baseline vs SVD",
            xaxis_title="K", yaxis_title="Precision",
            height=360,
        )
        st.plotly_chart(fig_prec, use_container_width=True)

    with col2:
        fig_rec = go.Figure()
        fig_rec.add_trace(go.Bar(name="Baseline", x=cmp["K"].astype(str),
                                  y=cmp["Recall_baseline"],
                                  marker_color=PALETTE[3]))
        fig_rec.add_trace(go.Bar(name="SVD", x=cmp["K"].astype(str),
                                  y=cmp["Recall_SVD"],
                                  marker_color=PALETTE[0]))
        fig_rec.update_layout(
            barmode="group",
            title="Recall@K: Baseline vs SVD",
            xaxis_title="K", yaxis_title="Recall",
            height=360,
        )
        st.plotly_chart(fig_rec, use_container_width=True)

    st.divider()

    st.subheader("Analisis del Resultado – Por que el SVD no supera al Baseline")
    col3, col4 = st.columns(2)
    with col3:
        st.error("""
**Hallazgo central de Sprint 2:**

El modelo SVD item-item obtiene menor Recall@10 que el baseline de popularidad.

**Causas identificadas:**
1. **Esparsidad extrema** (99.96%): el 97% de los clientes tiene solo 1 categoria comprada.
   El CF item-item asume que los usuarios tienen historial para calcular preferencias.

2. **Diversidad de categorias**: la siguiente compra de un cliente NO sigue
   necesariamente una logica de similitud de categorias. Los usuarios exploran.

3. **Dominio de popularidad**: las 10 categorias mas populares acaparan el 60%
   de todas las compras, por lo que el baseline es muy dificil de superar en recall global.
        """)

    with col4:
        st.success("""
**Lo que el SVD SI aporta:**

1. **Estructura latente visualizable**: el mapa 2D muestra clusters
   coherentes de categorias (ej. hogar, electronica, moda).

2. **Recomendaciones mas targetizadas** para clientes con 3+ compras:
   el SVD podria ser mas preciso que el baseline en ese segmento.

3. **Base para el modelo hibrido** de Sprint 3:
   SVD + popularidad + features RFM pueden superar al baseline.

4. **Pipeline reproducible establecido**: el pipeline corre mensualmente
   sin modificaciones, incorporando nuevos datos automaticamente.
        """)

    st.divider()

    # Evaluacion por periodo temporal
    st.subheader("Evaluacion por Periodo Temporal (Train / Val / Backtest)")
    st.dataframe(split_eval, use_container_width=True, hide_index=True)

    fig_split = px.bar(split_eval, x="Periodo", y="Recall@10",
                       color="Periodo", text="Recall@10",
                       title="Recall@10 del modelo SVD por periodo temporal",
                       color_discrete_sequence=PALETTE)
    fig_split.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig_split.update_layout(height=340, showlegend=False)
    st.plotly_chart(fig_split, use_container_width=True)

    st.caption("""
El Recall@10 en Backtest es mayor que en Validacion porque el modelo fue entrenado con
mas datos (hasta 2018-03) y el Backtest contiene clientes con mayor frecuencia de compra.
    """)

    st.divider()

    # Roadmap actualizado
    st.subheader("Roadmap de Mejora – Sprint 3")
    roadmap = pd.DataFrame({
        "Sprint": ["Sprint 1", "Sprint 2", "Sprint 3", "Sprint 4"],
        "Modelo": [
            "Baseline popularidad global",
            "SVD item-item (filtrado colaborativo)",
            "Hibrido: SVD + Popularidad + RFM segmentado",
            "API de recomendacion en tiempo real + MLflow",
        ],
        "Recall@10": ["0.6307", "0.1672", "Objetivo: >= 0.70", "Monitoreo continuo"],
        "Estado": ["Completado", "Completado", "Proximo", "Futuro"],
    })
    st.dataframe(roadmap, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 5 – SIMULACION MENSUAL
# ══════════════════════════════════════════════
with tab5:
    st.header("Simulacion Mensual – Pipeline Reproducible")

    st.subheader("Division Cronologica del Dataset")
    st.dataframe(splits["summary"], use_container_width=True, hide_index=True)

    # Pedidos por periodo
    train_ts = (splits["train"].groupby("order_ym")["order_id"]
                .nunique().reset_index().assign(Periodo="Train"))
    val_ts   = (splits["val"].groupby("order_ym")["order_id"]
                .nunique().reset_index().assign(Periodo="Validacion"))
    bt_ts    = (splits["backtest"].groupby("order_ym")["order_id"]
                .nunique().reset_index().assign(Periodo="Backtest"))
    all_ts   = pd.concat([train_ts, val_ts, bt_ts])
    all_ts   = all_ts[all_ts["order_ym"] != "NaT"]
    all_ts.columns = ["Mes", "Pedidos", "Periodo"]

    fig_ts = px.bar(all_ts, x="Mes", y="Pedidos", color="Periodo",
                    title="Pedidos mensuales por periodo (Train / Validacion / Backtest)",
                    color_discrete_map={"Train": PALETTE[0],
                                        "Validacion": PALETTE[1],
                                        "Backtest": PALETTE[2]})
    fig_ts.update_xaxes(tickangle=45)
    fig_ts.update_layout(height=380)
    st.plotly_chart(fig_ts, use_container_width=True)

    st.divider()

    # Simulacion acumulada mes a mes
    st.subheader("Evolucion de Metricas Mes a Mes (Reentrenamiento Acumulado)")

    if len(monthly) > 0:
        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Scatter(
            x=monthly["Mes"], y=monthly["Recall_SVD"],
            name="Recall@10 SVD", mode="lines+markers",
            line=dict(color=PALETTE[0], width=2),
        ))
        fig_monthly.add_trace(go.Scatter(
            x=monthly["Mes"], y=monthly["Recall_Baseline"],
            name="Recall@10 Baseline", mode="lines+markers",
            line=dict(color=PALETTE[3], width=2, dash="dash"),
        ))
        fig_monthly.update_layout(
            title="Recall@10 acumulado mes a mes (modelo reentrenado cada mes)",
            xaxis_title="Mes de corte",
            yaxis_title="Recall@10",
            height=380,
            legend=dict(x=0.7, y=0.2),
        )
        fig_monthly.update_xaxes(tickangle=45)
        st.plotly_chart(fig_monthly, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_users = px.area(monthly, x="Mes", y="Clientes_train",
                                title="Clientes acumulados en entrenamiento",
                                color_discrete_sequence=[PALETTE[2]],
                                labels={"Clientes_train": "Clientes"})
            fig_users.update_xaxes(tickangle=45)
            fig_users.update_layout(height=300)
            st.plotly_chart(fig_users, use_container_width=True)

        with col2:
            fig_evals = px.bar(monthly, x="Mes", y="Usuarios_eval",
                               title="Usuarios evaluados (con 2+ categorias)",
                               color_discrete_sequence=[PALETTE[1]],
                               labels={"Usuarios_eval": "Usuarios"})
            fig_evals.update_xaxes(tickangle=45)
            fig_evals.update_layout(height=300)
            st.plotly_chart(fig_evals, use_container_width=True)

        st.subheader("Tabla de simulacion mensual completa")
        st.dataframe(monthly.round(4), use_container_width=True, hide_index=True, height=300)
    else:
        st.warning("No hay suficientes datos para la simulacion mensual.")

    st.divider()

    st.subheader("Como funciona el Pipeline Mensual")
    st.info("""
**Cada mes, el pipeline ejecuta automaticamente:**

1. **Ingesta**: nuevos CSVs de pedidos, items, resenas y pagos del mes
2. **Limpieza**: aplica las mismas reglas (cancelados, nulos, outliers)
3. **Actualizacion de Master Table**: concatena con datos historicos
4. **Recomputo de features**: RFM actualizado con datos del nuevo mes
5. **Reentrenamiento SVD**: matriz usuario-categoria ampliada
6. **Exportacion**: modelo guardado en `models/rec_model_YYYYMM.pkl`
7. **Registro en MLflow**: Precision@10, Recall@10, n_users, timestamp

**Criterio de reentrenamiento:** siempre que haya ≥ 100 nuevos pedidos
    """)


# ══════════════════════════════════════════════
# TAB 6 – TARGET FINAL & METRICAS DE NEGOCIO
# ══════════════════════════════════════════════
with tab6:
    st.header("Target Final Confirmado & Metricas de Negocio")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Target Definitivo")
        st.success("""
**Variable objetivo:** `category_en`

La categoria de producto en ingles que el cliente comprara en su
proxima transaccion.

**Justificacion tecnica:**
- 73 categorias vs 33.000 productos → cardinalidad manejable
- Suficientes datos de co-compra para entrenamiento
- Evaluation realista con leave-one-out cronologico

**Justificacion de negocio:**
- Las campanas de email/push se disenan por categoria, no por SKU
- Permite recomendaciones incluso para productos sin historial
- Actualizable mensualmente con nuevos pedidos
        """)

    with col2:
        st.subheader("Metricas Tecnicas – Sprint 2")
        metricas_df = pd.DataFrame({
            "Metrica": [
                "Precision@10 Baseline",
                "Recall@10 Baseline",
                "Precision@10 SVD",
                "Recall@10 SVD",
                "Varianza explicada SVD (15 comp.)",
                "Esparsidad de la matriz",
                "Usuarios evaluados (LOO)",
            ],
            "Valor": [
                f"{cmp.loc[cmp['K']==10,'Precision_baseline'].values[0]:.4f}",
                f"{cmp.loc[cmp['K']==10,'Recall_baseline'].values[0]:.4f}",
                f"{cmp.loc[cmp['K']==10,'Precision_SVD'].values[0]:.4f}",
                f"{cmp.loc[cmp['K']==10,'Recall_SVD'].values[0]:.4f}",
                f"{sum(svd_model.explained_variance_ratio_)*100:.1f}%",
                f"{stats['sparsity_pct']:.2f}%",
                f"{cmp['K'].map(lambda k: None).iloc[0] or hist[hist['n_categories']>=2].shape[0]:,}",
            ],
        })
        st.dataframe(metricas_df, use_container_width=True, hide_index=True)

    st.divider()

    # KPIs negocio actualizados (dataset limpio)
    st.subheader("KPIs de Negocio – Dataset Limpio (Sprint 2)")
    from sprint1_eda import business_metrics
    bm = business_metrics(master)
    bm_raw = business_metrics(master_raw)

    kpi_cols = st.columns(6)
    kpis = [
        ("Pedidos", f"{bm['total_orders']:,}"),
        ("Clientes", f"{bm['total_customers']:,}"),
        ("Ticket medio (BRL)", str(bm['avg_order_value_brl'])),
        ("Satisfaccion media", f"{bm['avg_review_score']} / 5"),
        ("Tasa recompra", f"{bm['repeat_customer_rate']} %"),
        ("Entrega media (dias)", str(bm['avg_delivery_days'])),
    ]
    for col, (label, val) in zip(kpi_cols, kpis):
        col.metric(label, val)

    st.divider()

    st.subheader("Impacto de Negocio Proyectado – Sistema de Recomendacion")
    total_clients = bm["total_customers"]
    avg_ticket    = bm["avg_order_value_brl"]
    repeat_rate   = bm["repeat_customer_rate"] / 100

    scenarios = pd.DataFrame({
        "Escenario": ["Conservador", "Moderado", "Optimista"],
        "Lift recompra": ["+1%", "+3%", "+5%"],
        "Clientes adicionales": [
            int(total_clients * 0.01),
            int(total_clients * 0.03),
            int(total_clients * 0.05),
        ],
        "Ingresos adicionales mensuales (BRL)": [
            f"{int(total_clients * 0.01 * avg_ticket):,}",
            f"{int(total_clients * 0.03 * avg_ticket):,}",
            f"{int(total_clients * 0.05 * avg_ticket):,}",
        ],
    })
    st.dataframe(scenarios, use_container_width=True, hide_index=True)

    st.divider()

    # Conclusiones Sprint 2
    st.subheader("Conclusiones Sprint 2")
    st.info("""
**Logros del Sprint 2:**

1. Pipeline reproducible construido: 7 pasos modulares, ejecutable mensualmente
2. Feature Engineering completado: 15 variables RFM + 8 variables de categoria
3. Matriz de interacciones construida: 96K usuarios x 73 categorias (esparsidad 99.96%)
4. Modelo SVD entrenado: 15 componentes, 37.9% varianza explicada
5. Hallazgo clave: el SVD item-item puro no supera al baseline por la esparsidad extrema
6. Division temporal establecida: Train / Validacion / Backtest

**Decision para Sprint 3:**
Implementar modelo hibrido: popularidad global para clientes frios (1 compra)
+ SVD item-item para clientes recurrentes (2+ compras) + pesos ajustados por RFM.
Target `category_en` CONFIRMADO. Temporalidad: continua (batch diario / mensual).
    """)
