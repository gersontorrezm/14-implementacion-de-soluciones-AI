"""
Sprint 3 – Hiperparametrización y Modelo Final
Caso #8: Recomendación de Productos Personalizados
Dataset: Olist Brazilian E-Commerce | Maestría Data Science

Objetivo del sprint (según plan de trabajo):
  - Selección del modelo base (popularidad, co-compra, SVD, NMF, híbrido).
  - Hiperparametrización (GridSearchCV / RandomSearch).
  - Evaluación con validación cruzada y comparación de modelos.
  - Exportación del modelo final en formato pickle (.pkl) para ejecución mensual.
  - Documentar el pipeline de retraining mensual.

Métricas: Precision@k, Recall@k, F1@k, HitRate@k, Coverage (sistema de recomendación).
Target FINAL confirmado: `category_en` (siguiente categoría a recomendar).
"""

import json
import time
from pathlib import Path
from datetime import datetime
from itertools import combinations, product
from collections import Counter
import warnings

import numpy as np
import pandas as pd
import joblib

import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# Reutilizamos los bloques de Sprint 1 y Sprint 2
from sprint1_eda import (
    load_raw_tables,
    build_master_table,
    build_customer_history,
)
from sprint2_pipeline import (
    get_cleaning_report,
    build_interaction_matrix,
    build_temporal_splits,
)

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_VERSION = "3.0"
RANDOM_STATE = 42


# ═════════════════════════════════════════════════════════════
# 1. MODELO UNIFICADO DE RECOMENDACIÓN
# ═════════════════════════════════════════════════════════════

class CategoryRecommender:
    """
    Recomendador unificado de categorías (siguiente compra).

    Implementa 5 familias de modelos bajo una misma interfaz fit/recommend,
    lo que permite compararlos y tunearlos de forma homogénea:

      - "popularity" : top categorías más compradas (baseline).
      - "copurchase" : filtrado colaborativo item-item (similitud Jaccard).
      - "svd"        : TruncatedSVD sobre matriz usuario×categoría.
      - "nmf"        : factorización no negativa (NMF).
      - "hybrid"     : combinación ponderada CF(SVD) + popularidad (alpha).

    Hiperparámetros tuneables:
      n_components : nº de factores latentes (svd/nmf/hybrid).
      alpha        : peso del componente colaborativo vs popularidad (hybrid).
      n_iter       : iteraciones de TruncatedSVD.
      min_support  : soporte mínimo de co-compra (copurchase).
    """

    def __init__(self, method="hybrid", n_components=15, n_iter=10,
                 alpha=0.5, min_support=10, cf_method="copurchase",
                 random_state=RANDOM_STATE):
        self.method = method
        self.n_components = n_components
        self.n_iter = n_iter
        self.alpha = alpha
        self.min_support = min_support
        self.cf_method = cf_method     # backbone colaborativo del híbrido: "copurchase" | "svd"
        self.random_state = random_state

    # ── entrenamiento ────────────────────────────────────────
    def fit(self, master: pd.DataFrame):
        # Popularidad (siempre, sirve de fallback)
        pop = (master.groupby("category_en")["order_id"].nunique()
               .sort_values(ascending=False))
        self.popularity_ = pop
        self.pop_list_ = pop.index.tolist()
        self.pop_score_ = (pop / pop.max()).astype(float)   # 0..1

        # Matriz de interacciones usuario × categoría
        matrix, users, cats, u2i, c2i = build_interaction_matrix(master)
        self.cats_ = list(cats)
        self.n_users_ = matrix.shape[0]
        self.n_cats_ = matrix.shape[1]
        self.explained_variance_ratio_ = None

        max_k = max(1, min(self.n_components, matrix.shape[1] - 1))

        if self.method == "popularity":
            self.cat_sim_ = None

        elif self.method == "copurchase":
            self.cat_sim_ = self._build_copurchase_sim(master, cats)

        elif self.method == "svd":
            svd = TruncatedSVD(n_components=max_k, random_state=self.random_state,
                               n_iter=self.n_iter)
            svd.fit(matrix)
            self.model_ = svd
            self.explained_variance_ratio_ = svd.explained_variance_ratio_
            self.cat_factors_ = svd.components_.T
            self.cat_sim_ = pd.DataFrame(
                cosine_similarity(self.cat_factors_), index=cats, columns=cats)

        elif self.method == "nmf":
            nmf = NMF(n_components=max_k, init="nndsvda",
                      random_state=self.random_state, max_iter=300)
            nmf.fit(matrix.astype(float))
            self.model_ = nmf
            self.cat_factors_ = nmf.components_.T          # (n_cats, k)
            self.cat_sim_ = pd.DataFrame(
                cosine_similarity(self.cat_factors_), index=cats, columns=cats)

        elif self.method == "hybrid":
            # El híbrido combina un backbone colaborativo con la popularidad.
            if self.cf_method == "svd":
                svd = TruncatedSVD(n_components=max_k, random_state=self.random_state,
                                   n_iter=self.n_iter)
                svd.fit(matrix)
                self.model_ = svd
                self.explained_variance_ratio_ = svd.explained_variance_ratio_
                self.cat_factors_ = svd.components_.T
                self.cat_sim_ = pd.DataFrame(
                    cosine_similarity(self.cat_factors_), index=cats, columns=cats)
            else:   # backbone de co-compra (Jaccard) – mejor señal colaborativa
                self.cat_sim_ = self._build_copurchase_sim(master, cats)
        else:
            raise ValueError(f"Método desconocido: {self.method}")

        self.fitted_at_ = datetime.now().isoformat(timespec="seconds")
        return self

    def _build_copurchase_sim(self, master, cats):
        """Similitud  entre categorías co-compradas por un mismo cliente."""
        user_cats = (master.dropna(subset=["customer_unique_id", "category_en"])
                     .groupby("customer_unique_id")["category_en"].apply(set))
        pair, single = Counter(), Counter()
        for s in user_cats:
            for c in s:
                single[c] += 1
            for a, b in combinations(sorted(s), 2):
                pair[(a, b)] += 1

        sim = pd.DataFrame(0.0, index=list(cats), columns=list(cats))
        for (a, b), co in pair.items():
            if co < self.min_support:
                continue
            jacc = co / (single[a] + single[b] - co)
            sim.loc[a, b] = jacc
            sim.loc[b, a] = jacc
        np.fill_diagonal(sim.values, 1.0)
        return sim

    # ── inferencia ───────────────────────────────────────────
    def recommend(self, purchased_cats, k=10):
        """Top-k categorías recomendadas excluyendo las ya compradas."""
        purchased = set(purchased_cats)

        if self.method == "popularity" or self.cat_sim_ is None:
            return [c for c in self.pop_list_ if c not in purchased][:k]

        valid = [c for c in purchased_cats if c in self.cat_sim_.index]
        cf = pd.Series(0.0, index=self.cat_sim_.columns, dtype=float)
        if valid:
            cf = self.cat_sim_.loc[valid].sum(axis=0)
            if cf.max() > 0:
                cf = cf / cf.max()

        if self.method == "hybrid":
            pop = self.pop_score_.reindex(cf.index).fillna(0.0)
            score = self.alpha * cf + (1 - self.alpha) * pop
        else:
            score = cf

        score = score.drop(labels=[c for c in purchased if c in score.index],
                           errors="ignore")
        recs = score[score > 0].nlargest(k).index.tolist()

        # Completar con popularidad si faltan recomendaciones (cobertura)
        if len(recs) < k:
            extra = [c for c in self.pop_list_
                     if c not in purchased and c not in recs]
            recs += extra[:k - len(recs)]
        return recs[:k]

    def get_params_dict(self):
        return {
            "method": self.method,
            "cf_method": self.cf_method,
            "n_components": self.n_components,
            "n_iter": self.n_iter,
            "alpha": self.alpha,
            "min_support": self.min_support,
        }


# ═════════════════════════════════════════════════════════════
# 2. EVALUACIÓN (leave-one-out)  →  Precision@k / Recall@k / F1 / Coverage
# ═════════════════════════════════════════════════════════════

def evaluate_recommender(model: CategoryRecommender,
                         customer_history: pd.DataFrame, k=10) -> dict:
    """
    Evaluación leave-one-out sobre clientes con >=2 categorías distintas:
    se oculta la última categoría comprada y se verifica si aparece en el top-k.
    """
    multi = customer_history[customer_history["n_categories"] >= 2]
    precisions, recalls, hits_total = [], [], 0
    recommended_cats = set()

    for _, row in multi.iterrows():
        unique = list(dict.fromkeys(row["category_history"]))
        train_cats, test_cat = unique[:-1], unique[-1]
        recs = model.recommend(train_cats, k=k)
        recommended_cats.update(recs)
        hit = 1 if test_cat in recs else 0
        precisions.append(hit / k)
        recalls.append(float(hit))   # 1 sola categoría relevante
        hits_total += hit

    n = max(len(precisions), 1)
    p = float(np.mean(precisions)) if precisions else 0.0
    r = float(np.mean(recalls)) if recalls else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {
        f"Precision@{k}": round(p, 4),
        f"Recall@{k}": round(r, 4),
        f"F1@{k}": round(f1, 4),
        f"HitRate@{k}": round(hits_total / n, 4),
        "Coverage": round(len(recommended_cats) / max(len(model.cats_), 1), 4),
        "n_users": int(n),
    }


# ═════════════════════════════════════════════════════════════
# 3. VALIDACIÓN CRUZADA (K-Fold)
# ═════════════════════════════════════════════════════════════

def cross_validate_model(params: dict, master: pd.DataFrame,
                         customer_history: pd.DataFrame,
                         k=10, n_splits=4, random_state=RANDOM_STATE) -> pd.DataFrame:
    """
    K-Fold sobre clientes multi-categoría. En cada fold el modelo se ajusta
    SOLO con los clientes de entrenamiento (sin fuga del fold de test) y se
    evalúa leave-one-out sobre el fold retenido.
    """
    multi = (customer_history[customer_history["n_categories"] >= 2]
             .reset_index(drop=True))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []

    for fold, (tr_idx, te_idx) in enumerate(kf.split(multi), start=1):
        test_users = set(multi.loc[te_idx, "customer_unique_id"])
        train_master = master[~master["customer_unique_id"].isin(test_users)]
        model = CategoryRecommender(**params).fit(train_master)
        m = evaluate_recommender(model, multi.loc[te_idx], k=k)
        m["fold"] = fold
        rows.append(m)

    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════
# 4. GRID SEARCH y RANDOM SEARCH (con validación cruzada)
# ═════════════════════════════════════════════════════════════

def grid_search_cv(param_grid: dict, master: pd.DataFrame,
                   customer_history: pd.DataFrame,
                   k=10, n_splits=3) -> pd.DataFrame:
    """GridSearchCV: producto cartesiano del grid, CV en cada combinación."""
    keys = list(param_grid.keys())
    rows = []
    combos = list(product(*param_grid.values()))
    for i, combo in enumerate(combos, start=1):
        params = dict(zip(keys, combo))
        t0 = time.time()
        cv = cross_validate_model(params, master, customer_history,
                                  k=k, n_splits=n_splits)
        rows.append({
            **params,
            f"Precision@{k}_mean": round(cv[f"Precision@{k}"].mean(), 4),
            f"Recall@{k}_mean": round(cv[f"Recall@{k}"].mean(), 4),
            f"Recall@{k}_std": round(cv[f"Recall@{k}"].std(), 4),
            f"F1@{k}_mean": round(cv[f"F1@{k}"].mean(), 4),
            "Coverage_mean": round(cv["Coverage"].mean(), 4),
            "fit_time_s": round(time.time() - t0, 2),
        })
    res = pd.DataFrame(rows).sort_values(f"Recall@{k}_mean", ascending=False)
    return res.reset_index(drop=True)


def random_search_cv(param_distributions: dict, master: pd.DataFrame,
                     customer_history: pd.DataFrame, n_iter=8,
                     k=10, n_splits=3, random_state=RANDOM_STATE) -> pd.DataFrame:
    """RandomSearch: muestrea n_iter combinaciones aleatorias del espacio."""
    rng = np.random.default_rng(random_state)
    keys = list(param_distributions.keys())
    seen, rows = set(), []
    attempts = 0
    while len(rows) < n_iter and attempts < n_iter * 10:
        attempts += 1
        combo = tuple(rng.choice(param_distributions[key]) for key in keys)
        if combo in seen:
            continue
        seen.add(combo)
        params = {key: (int(v) if isinstance(v, np.integer) else
                        float(v) if isinstance(v, np.floating) else v)
                  for key, v in zip(keys, combo)}
        cv = cross_validate_model(params, master, customer_history,
                                  k=k, n_splits=n_splits)
        rows.append({
            **params,
            f"Precision@{k}_mean": round(cv[f"Precision@{k}"].mean(), 4),
            f"Recall@{k}_mean": round(cv[f"Recall@{k}"].mean(), 4),
            f"F1@{k}_mean": round(cv[f"F1@{k}"].mean(), 4),
            "Coverage_mean": round(cv["Coverage"].mean(), 4),
        })
    res = pd.DataFrame(rows).sort_values(f"Recall@{k}_mean", ascending=False)
    return res.reset_index(drop=True)


def optuna_search_cv(master: pd.DataFrame, customer_history: pd.DataFrame,
                     n_trials=20, k=10, n_splits=3,
                     random_state=RANDOM_STATE) -> dict:
    """
    Optimización bayesiana (TPE) con Optuna sobre el espacio de hiperparámetros.

    Ventajas frente a Grid/Random:
      - Espacio CONDICIONAL: n_components solo se muestrea si cf_method="svd".
      - alpha CONTINUO en [0, 1] (no discretizado).
      - Muestreo guiado (Tree-structured Parzen Estimator) en vez de ciego.

    Objetivo: maximizar Recall@k medio en validación cruzada K-Fold.
    Retorna dict con: study, best_params, results_df, importances.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        cf_method = trial.suggest_categorical("cf_method", ["copurchase", "svd"])
        alpha = trial.suggest_float("alpha", 0.0, 1.0)
        min_support = trial.suggest_int("min_support", 3, 30)
        params = {"method": "hybrid", "cf_method": cf_method,
                  "alpha": alpha, "min_support": min_support}
        # n_components solo es relevante con backbone SVD (espacio condicional)
        if cf_method == "svd":
            params["n_components"] = trial.suggest_int("n_components", 5, 40)

        cv = cross_validate_model(params, master, customer_history,
                                  k=k, n_splits=n_splits)
        trial.set_user_attr("precision", round(cv[f"Precision@{k}"].mean(), 4))
        trial.set_user_attr("coverage", round(cv["Coverage"].mean(), 4))
        return cv[f"Recall@{k}"].mean()

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name="recommender_tuning")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Tabla de resultados ordenada por Recall@k
    rows = []
    for t in study.trials:
        if t.value is None:
            continue
        rows.append({
            "trial": t.number,
            "cf_method": t.params.get("cf_method"),
            "alpha": round(t.params.get("alpha", np.nan), 4),
            "min_support": t.params.get("min_support"),
            "n_components": t.params.get("n_components", np.nan),
            f"Recall@{k}_mean": round(t.value, 4),
            f"Precision@{k}_mean": t.user_attrs.get("precision", np.nan),
            "Coverage_mean": t.user_attrs.get("coverage", np.nan),
        })
    results_df = (pd.DataFrame(rows)
                  .sort_values(f"Recall@{k}_mean", ascending=False)
                  .reset_index(drop=True))

    # Importancia de hiperparámetros (fanova)
    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception:
        importances = {}

    best_params = {"method": "hybrid", "cf_method": study.best_params["cf_method"],
                   "alpha": round(study.best_params["alpha"], 4),
                   "min_support": int(study.best_params["min_support"])}
    if study.best_params["cf_method"] == "svd":
        best_params["n_components"] = int(study.best_params["n_components"])

    return {
        "study": study,
        "best_params": best_params,
        "best_value": round(study.best_value, 4),
        "results": results_df,
        "importances": importances,
        "n_trials": len(study.trials),
    }


# ═════════════════════════════════════════════════════════════
# 5. COMPARACIÓN DE FAMILIAS DE MODELOS (selección del modelo base)
# ═════════════════════════════════════════════════════════════

def compare_model_families(master: pd.DataFrame,
                           customer_history: pd.DataFrame,
                           k=10, n_components=20, alpha=0.5) -> pd.DataFrame:
    """
    Entrena y evalúa (leave-one-out sobre todo el histórico) las 5 familias
    de modelos con parámetros por defecto, para justificar la selección base.
    """
    families = [
        ("Popularidad (baseline)", dict(method="popularity")),
        ("Co-compra item-item",    dict(method="copurchase", min_support=10)),
        ("SVD",                    dict(method="svd", n_components=n_components)),
        ("NMF",                    dict(method="nmf", n_components=n_components)),
        ("Híbrido (Cop+Pop)",      dict(method="hybrid", cf_method="copurchase",
                                        alpha=alpha, min_support=10)),
    ]
    rows = []
    for name, params in families:
        t0 = time.time()
        model = CategoryRecommender(**params).fit(master)
        m = evaluate_recommender(model, customer_history, k=k)
        rows.append({
            "Modelo": name,
            f"Precision@{k}": m[f"Precision@{k}"],
            f"Recall@{k}": m[f"Recall@{k}"],
            f"F1@{k}": m[f"F1@{k}"],
            "Coverage": m["Coverage"],
            "Tiempo_s": round(time.time() - t0, 2),
        })
    return pd.DataFrame(rows).sort_values(f"Recall@{k}", ascending=False).reset_index(drop=True)


# ═════════════════════════════════════════════════════════════
# 6. EVALUACIÓN TEMPORAL FINAL (backtest honesto)
# ═════════════════════════════════════════════════════════════

def evaluate_temporal_backtest(best_params: dict, master: pd.DataFrame,
                               k=10) -> dict:
    """
    Métrica DEFINITIVA: se ajusta el modelo con Train+Validación y se evalúa
    sobre el periodo de Backtest (datos nunca vistos), simulando producción.
    """
    splits = build_temporal_splits(master)
    train = pd.concat([splits["train"], splits["val"]], ignore_index=True)
    backtest = splits["backtest"]

    model = CategoryRecommender(**best_params).fit(train)
    train_hist = build_customer_history(train)
    truth = backtest.groupby("customer_unique_id")["category_en"].apply(set)

    eval_users = train_hist[train_hist["customer_unique_id"].isin(truth.index)]
    precisions, recalls = [], []
    for _, row in eval_users.iterrows():
        cust = row["customer_unique_id"]
        train_cats = list(set(row["category_history"]))
        relevant = truth.loc[cust] - set(train_cats)
        if not relevant:
            continue
        recs = set(model.recommend(train_cats, k=k))
        hits = len(recs & relevant)
        precisions.append(hits / k)
        recalls.append(hits / len(relevant))

    n = max(len(precisions), 1)
    return {
        f"Precision@{k}": round(float(np.mean(precisions)) if precisions else 0.0, 4),
        f"Recall@{k}": round(float(np.mean(recalls)) if recalls else 0.0, 4),
        "n_users": int(n),
        "periodo": "Backtest (2018-04 → 2018-10)",
    }


# ═════════════════════════════════════════════════════════════
# 7. FEATURE IMPORTANCE (para sistema de recomendación)
# ═════════════════════════════════════════════════════════════

def compute_factor_importance(model: CategoryRecommender) -> pd.DataFrame:
    """Importancia de cada factor latente = varianza explicada (SVD/híbrido)."""
    if model.explained_variance_ratio_ is None:
        return pd.DataFrame()
    evr = model.explained_variance_ratio_
    return pd.DataFrame({
        "Factor": [f"F{i+1}" for i in range(len(evr))],
        "Varianza_explicada": np.round(evr, 4),
        "Varianza_acumulada": np.round(np.cumsum(evr), 4),
    })


def compute_category_influence(model: CategoryRecommender, top=15) -> pd.DataFrame:
    """
    "Feature importance" análoga para recomendadores: influencia de cada
    categoría en las recomendaciones.
      - Backbone SVD/NMF: magnitud de cargas latentes (suma |loadings|).
      - Backbone co-compra: centralidad (suma de similitudes con otras categorías).
    Indica qué categorías estructuran más el modelo.
    """
    if hasattr(model, "cat_factors_"):
        influence = np.abs(model.cat_factors_).sum(axis=1)
        cats = model.cats_
    elif getattr(model, "cat_sim_", None) is not None:
        sim = model.cat_sim_.values.copy()
        np.fill_diagonal(sim, 0.0)
        influence = sim.sum(axis=1)
        cats = list(model.cat_sim_.index)
    else:
        return pd.DataFrame()
    df = pd.DataFrame({
        "Categoria": cats,
        "Influencia": np.round(influence, 4),
    }).sort_values("Influencia", ascending=False).reset_index(drop=True)
    return df.head(top)


# ═════════════════════════════════════════════════════════════
# 8. EXPORTACIÓN DEL MODELO FINAL (.pkl) + METADATOS
# ═════════════════════════════════════════════════════════════

def export_final_model(model: CategoryRecommender, metrics: dict,
                       model_dir: Path = MODELS_DIR,
                       filename="recommender_final.pkl") -> dict:
    """Serializa el modelo final con joblib y guarda metadatos JSON."""
    model_dir.mkdir(exist_ok=True)
    path = model_dir / filename
    joblib.dump(model, path)

    meta = {
        "model_version": MODEL_VERSION,
        "caso": "#8 Recomendación de Productos Personalizados",
        "target_final": "category_en",
        "algoritmo": model.method,
        "hiperparametros": model.get_params_dict(),
        "metricas_definitivas": metrics,
        "n_categorias": model.n_cats_,
        "n_usuarios_train": model.n_users_,
        "exportado_en": datetime.now().isoformat(timespec="seconds"),
        "artefacto": str(path.name),
    }
    meta_path = model_dir / "model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return {"model_path": str(path), "meta_path": str(meta_path), "metadata": meta}


def load_final_model(model_dir: Path = MODELS_DIR,
                     filename="recommender_final.pkl") -> CategoryRecommender:
    return joblib.load(model_dir / filename)


# ═════════════════════════════════════════════════════════════
# 9. PIPELINE DE RETRAINING MENSUAL (documentado y ejecutable)
# ═════════════════════════════════════════════════════════════

def monthly_retrain(best_params: dict, model_dir: Path = MODELS_DIR,
                    cutoff_ym: str = None) -> dict:
    """
    Pipeline de reentrenamiento mensual:
      1. Recarga las tablas base (actualizadas cada mes).
      2. Reconstruye y limpia la Master Table.
      3. (Opcional) filtra hasta cutoff_ym para simular el corte del mes.
      4. Reentrena el modelo final con los hiperparámetros óptimos.
      5. Versiona el artefacto con timestamp y registra métricas.
    """
    orders, customers, items, products, reviews, payments, sellers, cat_map = load_raw_tables()
    master_raw = build_master_table(orders, customers, items, products,
                                    reviews, payments, sellers, cat_map)
    _, master = get_cleaning_report(master_raw)

    if cutoff_ym:
        master = master[master["order_ym"] <= cutoff_ym]

    hist = build_customer_history(master)
    model = CategoryRecommender(**best_params).fit(master)
    metrics = evaluate_recommender(model, hist, k=10)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    versioned = f"recommender_{stamp}.pkl"
    info = export_final_model(model, metrics, model_dir=model_dir, filename=versioned)
    # también actualiza el "latest"
    joblib.dump(model, model_dir / "recommender_final.pkl")

    # Log de retraining (append)
    log_path = model_dir / "retrain_log.csv"
    log_row = pd.DataFrame([{
        "timestamp": stamp,
        "cutoff_ym": cutoff_ym or "full",
        "n_users": model.n_users_,
        "n_categorias": model.n_cats_,
        **metrics,
        "artefacto": versioned,
    }])
    if log_path.exists():
        log_row.to_csv(log_path, mode="a", header=False, index=False)
    else:
        log_row.to_csv(log_path, index=False)

    return {"metrics": metrics, "artefacto": versioned, **info}


# ═════════════════════════════════════════════════════════════
# 10. PIPELINE COMPLETO DEL SPRINT 3
# ═════════════════════════════════════════════════════════════

def run_sprint3_pipeline(k=10, n_splits=3, optuna_trials=20) -> dict:
    """
    Orquesta todo el Sprint 3 y retorna los artefactos para el dashboard:
      datos → comparación de familias → grid/random/optuna search → CV →
      backtest temporal → feature importance → exportación .pkl.
    """
    print("[1/9] Cargando y limpiando datos...")
    orders, customers, items, products, reviews, payments, sellers, cat_map = load_raw_tables()
    master_raw = build_master_table(orders, customers, items, products,
                                    reviews, payments, sellers, cat_map)
    _, master = get_cleaning_report(master_raw)
    hist = build_customer_history(master)

    print("[2/9] Comparando familias de modelos...")
    families = compare_model_families(master, hist, k=k)

    print("[3/9] GridSearchCV (alpha x min_support, backbone co-compra)...")
    grid = {
        "method": ["hybrid"],
        "cf_method": ["copurchase"],
        "alpha": [0.3, 0.5, 0.7, 0.9, 1.0],
        "min_support": [3, 5, 10],
    }
    t0 = time.time()
    grid_res = grid_search_cv(grid, master, hist, k=k, n_splits=n_splits)
    grid_time = round(time.time() - t0, 1)

    print("[4/9] RandomSearchCV (explora co-compra vs SVD)...")
    rand_dist = {
        "method": ["hybrid"],
        "cf_method": ["copurchase", "svd"],
        "alpha": [0.2, 0.3, 0.4, 0.5, 0.7],
        "n_components": [10, 20, 30],
        "min_support": [5, 10, 20],
    }
    t0 = time.time()
    rand_res = random_search_cv(rand_dist, master, hist, n_iter=8, k=k, n_splits=n_splits)
    rand_time = round(time.time() - t0, 1)

    print(f"[5/9] OptunaSearchCV (TPE bayesiano, {optuna_trials} trials)...")
    t0 = time.time()
    optuna_out = optuna_search_cv(master, hist, n_trials=optuna_trials,
                                  k=k, n_splits=n_splits)
    optuna_time = round(time.time() - t0, 1)
    optuna_res = optuna_out["results"]

    print("[6/9] Seleccionando mejor configuración entre los 3 métodos...")
    # Candidatos: mejor de cada método de tuning
    grid_best = grid_res.iloc[0]
    rand_best = rand_res.iloc[0]
    candidates = {
        "GridSearchCV": (float(grid_best[f"Recall@{k}_mean"]), {
            "method": "hybrid", "cf_method": str(grid_best["cf_method"]),
            "alpha": float(grid_best["alpha"]),
            "min_support": int(grid_best["min_support"])}),
        "RandomSearch": (float(rand_best[f"Recall@{k}_mean"]), {
            "method": "hybrid", "cf_method": str(rand_best["cf_method"]),
            "alpha": float(rand_best["alpha"]),
            "min_support": int(rand_best["min_support"]),
            **({"n_components": int(rand_best["n_components"])}
               if str(rand_best["cf_method"]) == "svd" else {})}),
        "Optuna (TPE)": (optuna_out["best_value"], optuna_out["best_params"]),
    }
    winner = max(candidates, key=lambda m: candidates[m][0])
    best_params = candidates[winner][1]

    # Tabla comparativa de los 3 métodos de tuning
    tuning_comparison = pd.DataFrame([
        {"Método": "GridSearchCV", f"Best Recall@{k}": candidates["GridSearchCV"][0],
         "Nº evaluaciones": len(grid_res), "Tiempo_s": grid_time,
         "Espacio": "discreto (cartesiano)", "Mejores_params": str(candidates["GridSearchCV"][1])},
        {"Método": "RandomSearch", f"Best Recall@{k}": candidates["RandomSearch"][0],
         "Nº evaluaciones": len(rand_res), "Tiempo_s": rand_time,
         "Espacio": "discreto (aleatorio)", "Mejores_params": str(candidates["RandomSearch"][1])},
        {"Método": "Optuna (TPE)", f"Best Recall@{k}": candidates["Optuna (TPE)"][0],
         "Nº evaluaciones": optuna_out["n_trials"], "Tiempo_s": optuna_time,
         "Espacio": "continuo + condicional", "Mejores_params": str(candidates["Optuna (TPE)"][1])},
    ]).sort_values(f"Best Recall@{k}", ascending=False).reset_index(drop=True)

    print(f"        → Ganador: {winner}  params={best_params}")

    print("[7/9] Entrenando modelo final + métricas definitivas...")
    final_model = CategoryRecommender(**best_params).fit(master)
    loo_metrics = evaluate_recommender(final_model, hist, k=k)
    backtest_metrics = evaluate_temporal_backtest(best_params, master, k=k)
    best_cv = cross_validate_model(best_params, master, hist, k=k, n_splits=n_splits)

    print("[8/9] Feature importance...")
    # SVD diagnóstico (solo para analizar la estructura latente / varianza explicada)
    svd_diag = CategoryRecommender(method="svd", n_components=20).fit(master)
    factor_imp = compute_factor_importance(svd_diag)
    cat_influence = compute_category_influence(final_model, top=15)

    print("[9/9] Exportando modelo final (.pkl)...")
    metrics_summary = {
        "leave_one_out": loo_metrics,
        "backtest_temporal": backtest_metrics,
        "cv_best": {
            f"Recall@{k}_mean": round(float(best_cv[f"Recall@{k}"].mean()), 4),
            f"Precision@{k}_mean": round(float(best_cv[f"Precision@{k}"].mean()), 4),
        },
        "tuning_winner": winner,
    }
    export_info = export_final_model(final_model, metrics_summary)

    print("Sprint 3 completado.")
    return {
        "master": master,
        "customer_history": hist,
        "family_comparison": families,
        "grid_results": grid_res,
        "random_results": rand_res,
        "optuna_results": optuna_res,
        "optuna_out": optuna_out,
        "tuning_comparison": tuning_comparison,
        "tuning_winner": winner,
        "best_params": best_params,
        "final_model": final_model,
        "loo_metrics": loo_metrics,
        "backtest_metrics": backtest_metrics,
        "factor_importance": factor_imp,
        "category_influence": cat_influence,
        "export_info": export_info,
        "k": k,
        "n_splits": n_splits,
    }


if __name__ == "__main__":
    art = run_sprint3_pipeline(k=10, n_splits=3)
    print("\n-- Comparacion de familias --")
    print(art["family_comparison"].to_string(index=False))
    print("\n-- Top GridSearch --")
    print(art["grid_results"].head().to_string(index=False))
    print("\n-- Top Optuna --")
    print(art["optuna_results"].head().to_string(index=False))
    print("\n-- Comparacion de metodos de tuning --")
    print(art["tuning_comparison"].to_string(index=False))
    print("\n-- Ganador:", art["tuning_winner"], "--")
    print("-- Mejores hiperparametros --")
    print(art["best_params"])
    print("\n-- Metricas LOO --", art["loo_metrics"])
    print("-- Metricas Backtest --", art["backtest_metrics"])
    print("\nModelo exportado en:", art["export_info"]["model_path"])
