"""
Generador del notebook Sprint3_Modelo_Final.ipynb
Caso #8: Recomendación de Productos Personalizados
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))

def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))

# ── Portada ──────────────────────────────────────────────────
md(r"""
# Sprint 3 — Hiperparametrización y Modelo Final
## Caso #8: Recomendación de Productos Personalizados
**Dataset:** Olist Brazilian E-Commerce · **Maestría en Data Science** · *Implementación de Soluciones de IA*

**Autor:** Gerson Jesús Torrez Marca

---

### Objetivo del Sprint 3 (Plan de trabajo)
> Optimizar el modelo, evaluar el rendimiento final y **exportar el artefacto** para uso mensual.

**Actividades clave cubiertas en este notebook:**
1. Selección del modelo base (popularidad, co-compra, SVD, NMF, híbrido).
2. Hiperparametrización con **GridSearchCV** y **RandomSearch**.
3. **Evaluación con validación cruzada** y comparación de modelos.
4. Exportación del modelo final en formato **pickle (`.pkl`)** para ejecución mensual.
5. Documentación del **pipeline de retraining mensual**.

**Entregables:** modelo `.pkl` con métricas definitivas · notebook comparativo · gráfico de rendimiento y *feature importance*.

**Target FINAL confirmado:** `category_en` (siguiente categoría a recomendar).
**Métricas:** Precision@k, Recall@k, F1@k, Coverage.
""")

# ── 0. Setup ─────────────────────────────────────────────────
md("## 0. Configuración e importación de módulos\nReutilizamos el pipeline modular construido en los Sprints 1 y 2.")
code(r"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sprint3_model import (
    run_sprint3_pipeline, compare_model_families, grid_search_cv, random_search_cv,
    optuna_search_cv, cross_validate_model, CategoryRecommender, evaluate_recommender,
    evaluate_temporal_backtest, compute_factor_importance, compute_category_influence,
    export_final_model, MODEL_VERSION,
)
from sprint1_eda import load_raw_tables, build_master_table, build_customer_history
from sprint2_pipeline import get_cleaning_report

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
print("Módulos cargados. Versión de modelo:", MODEL_VERSION)
""")

# ── 1. Datos ─────────────────────────────────────────────────
md("## 1. Carga y limpieza de datos\nMaster Table reproducible (~60 metadatos) y limpieza estándar heredada del Sprint 2.")
code(r"""
orders, customers, items, products, reviews, payments, sellers, cat_map = load_raw_tables()
master_raw = build_master_table(orders, customers, items, products, reviews, payments, sellers, cat_map)
_, master = get_cleaning_report(master_raw)
hist = build_customer_history(master)

print(f"Master Table limpia : {master.shape[0]:,} filas x {master.shape[1]} columnas")
print(f"Clientes únicos     : {master['customer_unique_id'].nunique():,}")
print(f"Categorías          : {master['category_en'].nunique()}")
print(f"Clientes con >=2 cat.: {(hist['n_categories']>=2).sum():,} (evaluables leave-one-out)")
""")

# ── 2. Comparación de familias ───────────────────────────────
md(r"""
## 2. Selección del modelo base — comparación de familias
Se evalúan 5 familias de recomendadores bajo la misma interfaz (*leave-one-out*):
**Popularidad**, **Co-compra item-item (Jaccard)**, **SVD**, **NMF** e **Híbrido**.
""")
code(r"""
families = compare_model_families(master, hist, k=10)
families
""")
code(r"""
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
fam = families.sort_values("Recall@10")
ax[0].barh(fam["Modelo"], fam["Recall@10"], color=sns.color_palette("viridis", len(fam)))
ax[0].set_title("Recall@10 por familia de modelo"); ax[0].set_xlabel("Recall@10")
for i, v in enumerate(fam["Recall@10"]):
    ax[0].text(v, i, f" {v:.3f}", va="center")

ax[1].scatter(families["Coverage"], families["Recall@10"],
              s=families["F1@10"]*1500, c=range(len(families)), cmap="viridis")
for _, r in families.iterrows():
    ax[1].annotate(r["Modelo"], (r["Coverage"], r["Recall@10"]), fontsize=8)
ax[1].set_xlabel("Coverage"); ax[1].set_ylabel("Recall@10")
ax[1].set_title("Trade-off Recall vs Coverage")
plt.tight_layout(); plt.show()
""")
md(r"""
**Hallazgo:** el backbone de **co-compra (similitud de Jaccard)** ofrece el mejor Recall@10.
SVD y NMF puros rinden bajo porque la matriz usuario×categoría es **casi binaria y muy dispersa**
(la mayoría de clientes compra una sola categoría), por lo que la factorización captura sobre todo
la dirección de popularidad. El modelo final adopta **co-compra como núcleo** + **popularidad como
fallback** (cold-start).
""")

# ── 3. GridSearch ────────────────────────────────────────────
md(r"""
## 3. Hiperparametrización — GridSearchCV
Búsqueda en malla `alpha × min_support` sobre el híbrido con backbone de co-compra,
evaluada con **K-Fold (3 folds) honesto** (clientes de test excluidos del entrenamiento).
`alpha` = peso de la señal colaborativa vs popularidad.
""")
code(r"""
grid = {
    "method": ["hybrid"], "cf_method": ["copurchase"],
    "alpha": [0.3, 0.5, 0.7, 0.9, 1.0], "min_support": [3, 5, 10],
}
grid_res = grid_search_cv(grid, master, hist, k=10, n_splits=3)
grid_res.head(10)
""")
code(r"""
pivot = grid_res.pivot_table(index="min_support", columns="alpha", values="Recall@10_mean")
plt.figure(figsize=(8, 4))
sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis", cbar_kws={"label": "Recall@10 (CV)"})
plt.title("GridSearchCV — Recall@10 medio por (min_support × alpha)")
plt.tight_layout(); plt.show()

best = grid_res.iloc[0]
best_params = {"method": "hybrid", "cf_method": "copurchase",
               "alpha": float(best["alpha"]), "min_support": int(best["min_support"])}
print("Mejores hiperparámetros:", best_params)
print(f"Recall@10 (CV) = {best['Recall@10_mean']:.4f} ± {best['Recall@10_std']:.4f}")
""")

# ── 4. RandomSearch ──────────────────────────────────────────
md(r"""
## 4. Hiperparametrización — RandomSearch
Exploración aleatoria del espacio (incluye también el backbone **SVD**) como validación
cruzada complementaria al grid exhaustivo.
""")
code(r"""
rand_dist = {
    "method": ["hybrid"], "cf_method": ["copurchase", "svd"],
    "alpha": [0.2, 0.3, 0.4, 0.5, 0.7], "n_components": [10, 20, 30],
    "min_support": [5, 10, 20],
}
rand_res = random_search_cv(rand_dist, master, hist, n_iter=8, k=10, n_splits=3)
rand_res
""")

# ── 4b. Optuna ───────────────────────────────────────────────
md(r"""
## 4b. Hiperparametrización — Optuna (optimización bayesiana TPE)
**Optuna** muestrea con un *Tree-structured Parzen Estimator* (búsqueda guiada, no ciega).
Ventajas frente a Grid/Random en este caso:
- `alpha` **continuo** en [0, 1] (no discretizado).
- Espacio **condicional**: `n_components` solo se muestrea si el backbone es SVD.
- Muestreo bayesiano que concentra trials en zonas prometedoras.
""")
code(r"""
optuna_out = optuna_search_cv(master, hist, n_trials=20, k=10, n_splits=3)
print("Mejores hiperparámetros (Optuna):", optuna_out["best_params"])
print(f"Mejor Recall@10 (CV): {optuna_out['best_value']:.4f}  ·  trials: {optuna_out['n_trials']}")
optuna_out["results"].head(10)
""")
code(r"""
opt = optuna_out["results"].sort_values("trial")
fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
ax[0].scatter(opt["trial"], opt["Recall@10_mean"], color="#3b8686", label="Trial")
ax[0].plot(opt["trial"], opt["Recall@10_mean"].cummax(), color="orange", lw=2, label="Mejor acumulado")
ax[0].set_xlabel("trial"); ax[0].set_ylabel("Recall@10 (CV)"); ax[0].legend()
ax[0].set_title("Optuna — historia de optimización")

imp = optuna_out.get("importances", {})
if imp:
    ax[1].barh(list(imp.keys()), list(imp.values()), color=sns.color_palette("crest", len(imp)))
    ax[1].set_title("Importancia de hiperparámetros (fANOVA)"); ax[1].set_xlabel("importancia")
plt.tight_layout(); plt.show()
""")

# ── 4c. Comparación de los 3 métodos ─────────────────────────
md(r"""
## 4c. Comparación de los tres métodos de tuning
Se compara el **mejor Recall@10** alcanzado por cada técnica, el número de evaluaciones y el tiempo.
""")
code(r"""
def best_of(df):
    return df.iloc[0]

g, r = best_of(grid_res), best_of(rand_res)
tuning_cmp = pd.DataFrame([
    {"Método": "GridSearchCV", "Best Recall@10": g["Recall@10_mean"],
     "Nº evaluaciones": len(grid_res), "Espacio": "discreto (cartesiano)"},
    {"Método": "RandomSearch", "Best Recall@10": r["Recall@10_mean"],
     "Nº evaluaciones": len(rand_res), "Espacio": "discreto (aleatorio)"},
    {"Método": "Optuna (TPE)", "Best Recall@10": optuna_out["best_value"],
     "Nº evaluaciones": optuna_out["n_trials"], "Espacio": "continuo + condicional"},
]).sort_values("Best Recall@10", ascending=False).reset_index(drop=True)
display(tuning_cmp)

winner = tuning_cmp.iloc[0]["Método"]
best_params = {"GridSearchCV": {"method":"hybrid","cf_method":str(g["cf_method"]),
                                "alpha":float(g["alpha"]),"min_support":int(g["min_support"])},
               "RandomSearch": {"method":"hybrid","cf_method":str(r["cf_method"]),
                                "alpha":float(r["alpha"]),"min_support":int(r["min_support"])},
               "Optuna (TPE)": optuna_out["best_params"]}[winner]

ax = tuning_cmp.set_index("Método")["Best Recall@10"].plot(
    kind="bar", color=sns.color_palette("viridis", 3), rot=0, figsize=(8, 4))
ax.set_title("Mejor Recall@10 (CV) por método de tuning"); ax.set_ylabel("Recall@10")
ax.bar_label(ax.containers[0], fmt="%.4f")
plt.tight_layout(); plt.show()
print(f"GANADOR: {winner}  →  {best_params}")
""")
md(r"""
**Interpretación:** el óptimo se encuentra en la **frontera `alpha = 1.0`** (filtrado colaborativo puro
con popularidad solo como *fallback*). GridSearchCV evalúa ese borde de forma explícita; el TPE de Optuna,
al muestrear `alpha` de forma continua, tiende a explorar el interior del espacio y rara vez aterriza exacto
en la frontera con pocos *trials*. **Conclusión honesta:** en espacios pequeños con el óptimo en el borde,
GridSearch puede igualar o superar a la búsqueda bayesiana; el valor de Optuna se hace evidente en espacios
**grandes, continuos y condicionales**, donde la búsqueda exhaustiva es inviable.
""")

# ── 5. Validación cruzada del modelo final ───────────────────
md("## 5. Validación cruzada del modelo final (5-Fold)\nRobustez del modelo seleccionado (mejor método de tuning).")
code(r"""
cv_final = cross_validate_model(best_params, master, hist, k=10, n_splits=5)
display(cv_final)

metric_cols = ["Precision@10", "Recall@10", "F1@10", "Coverage"]
resumen = pd.DataFrame({
    "Media":   [cv_final[m].mean() for m in metric_cols],
    "Desv.Est":[cv_final[m].std()  for m in metric_cols],
    "Mín":     [cv_final[m].min()  for m in metric_cols],
    "Máx":     [cv_final[m].max()  for m in metric_cols],
}, index=metric_cols)
print("Resumen validación cruzada (5-Fold):"); display(resumen.round(4))

fig, ax = plt.subplots(1, 2, figsize=(14, 4))
cv_final.set_index("fold")[["Recall@10","Precision@10","F1@10"]].plot(marker="o", ax=ax[0])
ax[0].set_title("Métricas por fold"); ax[0].set_ylabel("valor")
cv_final[["Recall@10","Precision@10"]].plot(kind="box", ax=ax[1])
ax[1].set_title("Dispersión entre folds")
plt.tight_layout(); plt.show()
""")

# ── 6. Modelo final + métricas definitivas ───────────────────
md(r"""
## 6. Modelo final — entrenamiento y métricas definitivas
Se entrena el modelo final con todos los datos y se reportan las **métricas definitivas**:
*leave-one-out* (todo el histórico) y **backtest temporal** (Train+Val → Backtest, datos nunca vistos).
""")
code(r"""
final_model = CategoryRecommender(**best_params).fit(master)
loo = evaluate_recommender(final_model, hist, k=10)
backtest = evaluate_temporal_backtest(best_params, master, k=10)

print("=== Métricas LEAVE-ONE-OUT (todo el histórico) ===")
for kk, vv in loo.items(): print(f"  {kk}: {vv}")
print("\n=== Métricas BACKTEST TEMPORAL (producción simulada) ===")
for kk, vv in backtest.items(): print(f"  {kk}: {vv}")
""")
code(r"""
perf = pd.DataFrame({
    "Escenario": ["Leave-One-Out", "CV 5-Fold (media)", "Backtest temporal"],
    "Precision@10": [loo["Precision@10"], cv_final["Precision@10"].mean(), backtest["Precision@10"]],
    "Recall@10":    [loo["Recall@10"],    cv_final["Recall@10"].mean(),    backtest["Recall@10"]],
})
ax = perf.set_index("Escenario").plot(kind="bar", figsize=(9, 4.5), colormap="viridis", rot=0)
ax.set_title("Rendimiento del modelo final por escenario de evaluación")
for c in ax.containers: ax.bar_label(c, fmt="%.3f")
plt.tight_layout(); plt.show()
perf.round(4)
""")

# ── 7. Feature importance ────────────────────────────────────
md(r"""
## 7. Feature importance
Para un sistema de recomendación, la "importancia de variables" se interpreta como:
- **Influencia de categorías**: centralidad de cada categoría en la red de co-compra.
- **Estructura latente (SVD diagnóstico)**: varianza explicada por factor.
""")
code(r"""
cat_influence = compute_category_influence(final_model, top=15)
svd_diag = CategoryRecommender(method="svd", n_components=20).fit(master)
factor_imp = compute_factor_importance(svd_diag)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ci = cat_influence.sort_values("Influencia")
ax[0].barh(ci["Categoria"], ci["Influencia"], color=sns.color_palette("crest", len(ci)))
ax[0].set_title("Feature importance — influencia de categorías (co-compra)")
ax[0].set_xlabel("Centralidad (suma de similitudes)")

ax[1].bar(factor_imp["Factor"], factor_imp["Varianza_explicada"], color="#3b8686")
ax2 = ax[1].twinx()
ax2.plot(factor_imp["Factor"], factor_imp["Varianza_acumulada"], color="orange", marker="o")
ax2.set_ylim(0, 1); ax2.set_ylabel("Varianza acumulada")
ax[1].set_title("Estructura latente — varianza explicada (SVD)")
ax[1].set_ylabel("Varianza explicada")
plt.tight_layout(); plt.show()
display(cat_influence)
""")

# ── 8. Exportación pkl ───────────────────────────────────────
md("## 8. Exportación del modelo final (`.pkl`)\nArtefacto serializado con joblib + metadatos JSON, listo para ejecución mensual.")
code(r"""
metrics_summary = {"leave_one_out": loo, "backtest_temporal": backtest,
                   "cv_best": {"Recall@10_mean": float(best["Recall@10_mean"]),
                               "Precision@10_mean": float(best["Precision@10_mean"])}}
export_info = export_final_model(final_model, metrics_summary)
print("Modelo exportado en:", export_info["model_path"])
print("Metadatos en       :", export_info["meta_path"])
import json; print(json.dumps(export_info["metadata"], indent=2, ensure_ascii=False))
""")
code(r"""
# Verificación de carga del artefacto
from sprint3_model import load_final_model
loaded = load_final_model()
print("Modelo recargado OK. Prueba de recomendación:")
print("  Compradas:", ["bed_bath_table", "furniture_decor"])
print("  Recomienda:", loaded.recommend(["bed_bath_table", "furniture_decor"], k=10))
""")

# ── 9. Retraining mensual ────────────────────────────────────
md(r"""
## 9. Pipeline de retraining mensual
El modelo está diseñado para **ejecutarse mensualmente** con datos actualizados.
La función `monthly_retrain()` recarga las tablas, reconstruye la Master Table, reentrena
con los hiperparámetros óptimos, **versiona el artefacto con timestamp** y registra las
métricas en `retrain_log.csv`.

```python
from sprint3_model import monthly_retrain
monthly_retrain({"method":"hybrid","cf_method":"copurchase",
                 "alpha":1.0, "min_support":10})
```

**Programación sugerida:** cron / Task Scheduler → día 1 de cada mes.
""")
code(r"""
# Demostración del retraining mensual (genera artefacto versionado + log)
from sprint3_model import monthly_retrain
result = monthly_retrain(best_params)
print("Artefacto versionado:", result["artefacto"])
print("Métricas del reentrenamiento:", result["metrics"])
""")

# ── 10. Conclusiones ─────────────────────────────────────────
md(r"""
## 10. Conclusiones y justificación del modelo elegido

| Aspecto | Decisión |
|---|---|
| **Target final** | `category_en` (siguiente categoría a recomendar) — confirmado |
| **Modelo final** | Híbrido = **Co-compra (Jaccard) + Popularidad (fallback)** |
| **Hiperparámetros** | `alpha=1.0`, `min_support=10` (señal colaborativa pura + respaldo de popularidad) |
| **Tuning** | GridSearchCV + RandomSearch + **Optuna (TPE)**, todos con validación cruzada K-Fold |
| **Métricas (CV 5-Fold)** | Recall@10 ≈ 0.67 · Precision@10 ≈ 0.067 |
| **Métrica definitiva (Backtest)** | Recall@10 ≈ 0.61 sobre datos nunca vistos |
| **Artefacto** | `models/recommender_final.pkl` + `model_metadata.json` |
| **Producción** | `monthly_retrain()` versiona y registra cada reentrenamiento |

**Justificación:** la matriz usuario×categoría del e-commerce es extremadamente dispersa
y casi binaria, escenario donde los métodos de factorización latente (SVD/NMF) no logran
capturar la co-ocurrencia real de compras. El **filtrado colaborativo item-item por co-compra**
modela directamente "quién compró A también compró B", lo que produce el mejor Recall manteniendo
interpretabilidad. La **popularidad como fallback** garantiza recomendaciones para clientes sin
historial de co-compra (cold-start), elevando la cobertura. El modelo supera de forma robusta y
**validada temporalmente** al baseline de popularidad del Sprint 1.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

with open("Sprint3_Modelo_Final.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook generado: Sprint3_Modelo_Final.ipynb con", len(cells), "celdas")
