"""
Dashboard Sprint 3 – Hiperparametrización y Modelo Final
Caso #8: Recomendación de Productos Personalizados
Olist Brazilian E-Commerce | Maestría Data Science

Ejecutar:  streamlit run dashboard_sprint3.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sprint3_model import (
    run_sprint3_pipeline,
    cross_validate_model,
    CategoryRecommender,
    MODEL_VERSION,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sprint 3 · Modelo Final · Recomendación Olist",
    page_icon="3",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = px.colors.qualitative.Bold
K = 10


# ─────────────────────────────────────────────
# CACHE – pipeline completo (se ejecuta una vez)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Ejecutando pipeline Sprint 3 (tuning + CV, ~3 min)...")
def load_artifacts():
    art = run_sprint3_pipeline(k=K, n_splits=3)
    # Validación cruzada detallada del modelo final (para el tab de CV)
    art["cv_final"] = cross_validate_model(
        art["best_params"], art["master"], art["customer_history"],
        k=K, n_splits=5)
    return art


art = load_artifacts()

master       = art["master"]
hist         = art["customer_history"]
families     = art["family_comparison"]
grid_res     = art["grid_results"]
rand_res     = art["random_results"]
optuna_res   = art["optuna_results"]
optuna_out   = art["optuna_out"]
tuning_cmp   = art["tuning_comparison"]
tuning_winner= art["tuning_winner"]
best_params  = art["best_params"]
final_model  = art["final_model"]
loo_metrics  = art["loo_metrics"]
backtest     = art["backtest_metrics"]
factor_imp   = art["factor_importance"]
cat_influence= art["category_influence"]
export_info  = art["export_info"]
cv_final     = art["cv_final"]


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("Sprint 3")
    st.caption("Caso #8 · Recomendación de Productos")
    st.caption("Gerson Jesus Torrez Marca")
    st.divider()
    st.markdown(f"**Versión modelo:** {MODEL_VERSION}")
    st.markdown(f"**Algoritmo final:** Híbrido (Co-compra + Popularidad)")
    st.markdown(f"**Target FINAL:** `category_en`")
    st.divider()
    st.markdown("**Hiperparámetros óptimos**")
    st.json(best_params)
    st.divider()
    st.markdown("**Métricas definitivas**")
    st.metric("Recall@10 (CV)", f"{tuning_cmp['Best Recall@10'].max():.4f}")
    st.metric("Recall@10 (Backtest)", f"{backtest['Recall@10']:.4f}")
    st.metric("Precision@10 (LOO)", f"{loo_metrics['Precision@10']:.4f}")
    st.divider()
    st.markdown(f"**Tuning ganador:** {tuning_winner}")
    st.caption("GridSearchCV · RandomSearch · Optuna (TPE)")
    st.caption("Plan: Sprint 3 – Hiperparametrización y modelo final")


st.title("Sprint 3 · Hiperparametrización y Modelo Final")
st.markdown(
    "Optimización del recomendador, **comparación de modelos con validación cruzada**, "
    "selección de hiperparámetros (**GridSearchCV / RandomSearch / Optuna**), evaluación definitiva "
    "(backtest temporal) y exportación del artefacto **`.pkl`** para ejecución mensual."
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1 · Selección de Modelo",
    "2 · Hiperparametrización",
    "3 · Validación Cruzada",
    "4 · Modelo Final & Feature Importance",
    "5 · Retraining Mensual",
    "6 · Demo de Recomendación",
])


# ══════════════════════════════════════════════
# TAB 1 – SELECCIÓN DE MODELO
# ══════════════════════════════════════════════
with tab1:
    st.header("Comparación de familias de modelos")
    st.markdown(
        "Se evaluaron 5 familias de recomendadores bajo la misma interfaz "
        "(*leave-one-out* sobre clientes con ≥2 categorías). El criterio principal "
        "es **Recall@10** (cobertura del acierto), complementado con Precision@10, F1 y Coverage."
    )

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.bar(
            families.sort_values("Recall@10"),
            x="Recall@10", y="Modelo", orientation="h",
            color="Modelo", color_discrete_sequence=PALETTE,
            text="Recall@10", title="Recall@10 por familia de modelo",
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.scatter(
            families, x="Coverage", y="Recall@10",
            size="F1@10", color="Modelo", text="Modelo",
            color_discrete_sequence=PALETTE,
            title="Recall vs Coverage (trade-off)",
        )
        fig2.update_traces(textposition="top center")
        fig2.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(families, use_container_width=True, hide_index=True)

    st.success(
        "**Decisión:** el backbone colaborativo de **co-compra (similitud de Jaccard)** "
        "es la señal más fuerte (Recall@10 ≈ 0.69). El modelo final lo adopta como núcleo "
        "y usa la **popularidad como fallback** para clientes sin co-compras (cold-start), "
        "maximizando cobertura sin sacrificar acierto. SVD/NMF puros rinden bajo porque la "
        "matriz usuario×categoría es casi binaria y muy dispersa."
    )


# ══════════════════════════════════════════════
# TAB 2 – HIPERPARAMETRIZACIÓN
# ══════════════════════════════════════════════
with tab2:
    st.header("Hiperparametrización con validación cruzada")
    st.markdown(
        "Se comparan **tres técnicas de tuning** sobre el mismo objetivo "
        "(maximizar Recall@10 en K-Fold honesto):\n"
        "- **GridSearchCV** — producto cartesiano `alpha × min_support` (espacio discreto).\n"
        "- **RandomSearch** — muestreo aleatorio (incluye backbone SVD).\n"
        "- **Optuna (TPE)** — optimización bayesiana, `alpha` **continuo** + espacio "
        "**condicional** (`n_components` solo si backbone = SVD)."
    )

    # ── Comparación de los 3 métodos ──
    st.subheader("Comparación de métodos de tuning")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.dataframe(tuning_cmp, use_container_width=True, hide_index=True)
    with c2:
        figc = px.bar(tuning_cmp, x="Método", y="Best Recall@10",
                      color="Método", text="Best Recall@10",
                      color_discrete_sequence=PALETTE,
                      title="Mejor Recall@10 por método")
        figc.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        figc.update_layout(showlegend=False, height=320,
                           yaxis_range=[0, tuning_cmp["Best Recall@10"].max()*1.15])
        st.plotly_chart(figc, use_container_width=True)
    st.success(
        f"**Ganador: {tuning_winner}.** El óptimo está en la frontera `alpha = 1.0` "
        "(filtrado colaborativo puro). GridSearch la evalúa explícitamente; el TPE de Optuna, "
        "al muestrear `alpha` continuo, tiende a explorar el interior y rara vez aterriza exacto "
        "en la frontera. En espacios pequeños con óptimo en el borde, el grid puede igualar o "
        "superar a la búsqueda bayesiana — cuyo valor real aparece en espacios grandes/continuos."
    )

    st.divider()
    tg, tr, to = st.tabs(["GridSearchCV", "RandomSearch", "Optuna (TPE)"])

    # ── GridSearchCV ──
    with tg:
        try:
            pivot = grid_res.pivot_table(
                index="min_support", columns="alpha",
                values="Recall@10_mean", aggfunc="mean")
            figh = px.imshow(
                pivot, text_auto=".4f", aspect="auto",
                color_continuous_scale="Viridis",
                labels=dict(color="Recall@10"),
                title="Recall@10 (CV) — alpha × min_support",
            )
            figh.update_layout(height=350)
            st.plotly_chart(figh, use_container_width=True)
        except Exception:
            pass
        st.dataframe(grid_res, use_container_width=True, hide_index=True)
        best = grid_res.iloc[0]
        st.info(
            f"Mejor grid: alpha = **{best['alpha']}**, min_support = "
            f"**{int(best['min_support'])}** → Recall@10 = **{best['Recall@10_mean']:.4f}** "
            f"(± {best['Recall@10_std']:.4f})."
        )

    # ── RandomSearch ──
    with tr:
        st.dataframe(rand_res, use_container_width=True, hide_index=True)
        st.caption("Muestreo aleatorio de 8 configuraciones, incluyendo el backbone SVD.")

    # ── Optuna ──
    with to:
        st.markdown("**Historia de optimización** (Recall@10 por trial y mejor acumulado)")
        opt_sorted = optuna_res.sort_values("trial")
        figo = go.Figure()
        figo.add_trace(go.Scatter(
            x=opt_sorted["trial"], y=opt_sorted["Recall@10_mean"],
            mode="markers", name="Trial", marker=dict(color=PALETTE[0], size=8)))
        figo.add_trace(go.Scatter(
            x=opt_sorted["trial"], y=opt_sorted["Recall@10_mean"].cummax(),
            mode="lines", name="Mejor acumulado", line=dict(color=PALETTE[1], width=3)))
        figo.update_layout(height=320, xaxis_title="trial", yaxis_title="Recall@10 (CV)")
        st.plotly_chart(figo, use_container_width=True)

        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Importancia de hiperparámetros** (fANOVA)")
            imp = optuna_out.get("importances", {})
            if imp:
                imp_df = (pd.DataFrame({"Hiperparámetro": list(imp.keys()),
                                        "Importancia": list(imp.values())})
                          .sort_values("Importancia"))
                figi = px.bar(imp_df, x="Importancia", y="Hiperparámetro",
                              orientation="h", color="Importancia",
                              color_continuous_scale="Tealgrn")
                figi.update_layout(height=260, coloraxis_showscale=False)
                st.plotly_chart(figi, use_container_width=True)
            else:
                st.caption("No disponible.")
        with cc2:
            st.markdown("**Distribución alpha vs Recall@10**")
            figa = px.scatter(optuna_res, x="alpha", y="Recall@10_mean",
                              color="cf_method", size="min_support",
                              color_discrete_sequence=PALETTE)
            figa.update_layout(height=260)
            st.plotly_chart(figa, use_container_width=True)
        st.dataframe(optuna_res, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 3 – VALIDACIÓN CRUZADA
# ══════════════════════════════════════════════
with tab3:
    st.header("Validación cruzada del modelo final (5-Fold)")
    st.markdown(
        "Robustez del modelo seleccionado mediante K-Fold sobre clientes. "
        "En cada fold el modelo se ajusta **sin** los clientes de prueba (sin fuga)."
    )

    metric_cols = [f"Precision@{K}", f"Recall@{K}", f"F1@{K}", "Coverage"]
    summary = pd.DataFrame({
        "Métrica": metric_cols,
        "Media":   [cv_final[m].mean() for m in metric_cols],
        "Desv.Est.": [cv_final[m].std() for m in metric_cols],
        "Mín":     [cv_final[m].min() for m in metric_cols],
        "Máx":     [cv_final[m].max() for m in metric_cols],
    }).round(4)

    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(summary, use_container_width=True, hide_index=True)
    with c2:
        figb = go.Figure()
        figb.add_trace(go.Box(y=cv_final[f"Recall@{K}"], name="Recall@10",
                              marker_color=PALETTE[0], boxpoints="all"))
        figb.add_trace(go.Box(y=cv_final[f"Precision@{K}"], name="Precision@10",
                              marker_color=PALETTE[1], boxpoints="all"))
        figb.update_layout(title="Dispersión por fold", height=320)
        st.plotly_chart(figb, use_container_width=True)

    figl = px.line(
        cv_final, x="fold", y=[f"Recall@{K}", f"Precision@{K}", f"F1@{K}"],
        markers=True, title="Métricas por fold",
        color_discrete_sequence=PALETTE,
    )
    figl.update_layout(height=320, yaxis_title="valor", xaxis_title="fold")
    st.plotly_chart(figl, use_container_width=True)

    st.dataframe(cv_final, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 4 – MODELO FINAL & FEATURE IMPORTANCE
# ══════════════════════════════════════════════
with tab4:
    st.header("Modelo final · métricas definitivas y feature importance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision@10 (LOO)", f"{loo_metrics['Precision@10']:.4f}")
    c2.metric("Recall@10 (LOO)", f"{loo_metrics['Recall@10']:.4f}")
    c3.metric("Recall@10 (Backtest)", f"{backtest['Recall@10']:.4f}",
              help=backtest["periodo"])
    c4.metric("Coverage", f"{loo_metrics['Coverage']:.4f}")

    st.markdown("#### Comparación de rendimiento por escenario de evaluación")
    perf = pd.DataFrame({
        "Escenario": ["Leave-One-Out (todo)", "CV 5-Fold (media)", "Backtest temporal"],
        "Precision@10": [loo_metrics["Precision@10"],
                         cv_final[f"Precision@{K}"].mean(),
                         backtest["Precision@10"]],
        "Recall@10": [loo_metrics["Recall@10"],
                      cv_final[f"Recall@{K}"].mean(),
                      backtest["Recall@10"]],
    }).round(4)
    figp = px.bar(perf.melt(id_vars="Escenario", var_name="Métrica", value_name="Valor"),
                  x="Escenario", y="Valor", color="Métrica", barmode="group",
                  text="Valor", color_discrete_sequence=PALETTE,
                  title="Rendimiento del modelo final")
    figp.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    figp.update_layout(height=380)
    st.plotly_chart(figp, use_container_width=True)

    st.divider()
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("#### Feature Importance — Influencia de categorías")
        st.caption("Centralidad de cada categoría en la red de co-compra "
                   "(suma de similitudes). Indica qué categorías estructuran las recomendaciones.")
        figi = px.bar(cat_influence.sort_values("Influencia"),
                      x="Influencia", y="Categoria", orientation="h",
                      color="Influencia", color_continuous_scale="Tealgrn",
                      height=480)
        figi.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(figi, use_container_width=True)
    with cc2:
        st.markdown("#### Estructura latente (SVD diagnóstico)")
        st.caption("Varianza explicada por factor latente — cuántos factores "
                   "capturan la estructura de compra (análisis complementario).")
        if not factor_imp.empty:
            figv = go.Figure()
            figv.add_trace(go.Bar(x=factor_imp["Factor"],
                                  y=factor_imp["Varianza_explicada"],
                                  name="Por factor", marker_color=PALETTE[2]))
            figv.add_trace(go.Scatter(x=factor_imp["Factor"],
                                      y=factor_imp["Varianza_acumulada"],
                                      name="Acumulada", yaxis="y2",
                                      mode="lines+markers", line=dict(color=PALETTE[3])))
            figv.update_layout(
                height=480, yaxis=dict(title="Varianza explicada"),
                yaxis2=dict(title="Acumulada", overlaying="y", side="right",
                            range=[0, 1]),
                legend=dict(orientation="h", y=1.1))
            st.plotly_chart(figv, use_container_width=True)

    st.divider()
    st.markdown("#### Artefacto exportado (`.pkl`)")
    cexp1, cexp2 = st.columns([1, 1])
    with cexp1:
        st.code(export_info["model_path"], language="text")
        st.caption("Modelo serializado con joblib — listo para ejecución mensual.")
    with cexp2:
        st.json(export_info["metadata"])


# ══════════════════════════════════════════════
# TAB 5 – RETRAINING MENSUAL
# ══════════════════════════════════════════════
with tab5:
    st.header("Pipeline de retraining mensual")
    st.markdown(
        "El modelo se reentrena mensualmente con las tablas base actualizadas. "
        "El artefacto se versiona con *timestamp* y se registra cada ejecución."
    )

    st.code("""
PIPELINE DE RETRAINING MENSUAL  —  monthly_retrain()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [1] load_raw_tables()            ← recarga 9 CSV actualizados del mes
  [2] build_master_table()         ← join + variables derivadas (~60 cols)
  [3] get_cleaning_report()        ← limpieza estándar reproducible
  [4] CategoryRecommender(**best)  ← reajuste con hiperparámetros óptimos
        method=hybrid · cf=copurchase · alpha=1.0 · min_support=10
  [5] evaluate_recommender()       ← Precision@10 / Recall@10 / Coverage
  [6] export_final_model()         ← recommender_YYYYMMDD_HHMM.pkl  (+ latest)
                                     + model_metadata.json
  [7] append → retrain_log.csv     ← trazabilidad de cada reentrenamiento

  Programación sugerida: cron / Task Scheduler  →  día 1 de cada mes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """, language="text")

    st.markdown("**Ejecución mensual (línea de comando):**")
    st.code(
        'python -c "from sprint3_model import monthly_retrain; '
        "monthly_retrain({'method':'hybrid','cf_method':'copurchase',"
        "'alpha':1.0,'min_support':10})\"",
        language="bash",
    )

    # Mostrar log de retraining si existe
    from pathlib import Path
    log_path = Path("models/retrain_log.csv")
    if log_path.exists():
        st.markdown("#### Historial de reentrenamientos (`retrain_log.csv`)")
        st.dataframe(pd.read_csv(log_path), use_container_width=True, hide_index=True)
    else:
        st.info("Aún no se ha registrado ningún reentrenamiento versionado. "
                "Ejecuta `monthly_retrain(...)` para generar el primer registro.")


# ══════════════════════════════════════════════
# TAB 6 – DEMO DE RECOMENDACIÓN
# ══════════════════════════════════════════════
with tab6:
    st.header("Demo interactiva del recomendador")
    st.markdown("Selecciona categorías ya compradas por un cliente y obtén "
                "las **top-10 recomendaciones** del modelo final.")

    all_cats = sorted(final_model.cats_)
    colA, colB = st.columns([2, 1])
    with colA:
        chosen = st.multiselect(
            "Categorías compradas por el cliente:",
            options=all_cats,
            default=all_cats[:2] if len(all_cats) >= 2 else all_cats,
        )
    with colB:
        topk = st.slider("Número de recomendaciones (k)", 3, 20, 10)

    if chosen:
        recs = final_model.recommend(chosen, k=topk)
        st.markdown("#### Recomendaciones")
        rec_df = pd.DataFrame({
            "Ranking": range(1, len(recs) + 1),
            "Categoría recomendada": recs,
        })
        st.dataframe(rec_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Selecciona al menos una categoría para generar recomendaciones.")

    st.divider()
    st.markdown("#### Probar con un cliente real del dataset")
    multi_users = hist[hist["n_categories"] >= 2]
    if len(multi_users) > 0:
        sample_user = st.selectbox(
            "Cliente (con ≥2 categorías):",
            options=multi_users["customer_unique_id"].head(200).tolist(),
        )
        urow = hist[hist["customer_unique_id"] == sample_user].iloc[0]
        ucats = list(dict.fromkeys(urow["category_history"]))
        st.write("**Historial de categorías:**", ucats)
        urecs = final_model.recommend(ucats, k=10)
        st.write("**Recomendaciones (top-10):**", urecs)
