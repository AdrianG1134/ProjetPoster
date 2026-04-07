#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise SystemExit(
        "xgboost n'est pas installe. Lance: pip install xgboost scikit-learn"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classification des cultures avec XGBoost a partir d'indices Sentinel-2"
    )
    parser.add_argument(
        "--indices-csv",
        default="data/s2_herault_2024_full/indices_parcelles_2024-01-01_2024-12-31_win5d.csv",
        help="CSV long des indices par parcelle",
    )
    parser.add_argument("--labels-csv", default="export.csv", help="CSV RPG (verite terrain)")
    parser.add_argument("--id-col", default="ID_PARCEL", help="Nom de la colonne identifiant parcelle")
    parser.add_argument("--target-col", default="CODE_CULTU", help="Nom de la colonne cible")
    parser.add_argument(
        "--indices", nargs="+", default=["NDVI", "EVI", "NDMI", "NDWI"], help="Indices a conserver"
    )
    parser.add_argument("--cloud-max", type=float, default=40.0, help="Seuil max cloud_scene")
    parser.add_argument("--px-count-min", type=int, default=5, help="Seuil min de pixels valides")
    parser.add_argument(
        "--min-class-count",
        type=int,
        default=200,
        help="Conserver les classes avec au moins N parcelles",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion du test set")
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Proportion de validation interne prelevee sur le train",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Graine aleatoire")
    parser.add_argument(
        "--augment",
        choices=["none", "smote", "borderline_smote"],
        default="none",
        help="Augmentation du train set (apres split): none, smote, borderline_smote",
    )
    parser.add_argument(
        "--smote-k-neighbors",
        type=int,
        default=5,
        help="Nombre de voisins pour SMOTE/BorderlineSMOTE",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Nombre de folds pour validation croisee stratifiee (>=2, 0 pour desactiver)",
    )
    parser.add_argument(
        "--tune-n-iter",
        type=int,
        default=15,
        help="Nombre d'iterations RandomizedSearchCV (0 pour desactiver)",
    )
    parser.add_argument(
        "--tune-verbose",
        type=int,
        default=1,
        help="Verbosite de RandomizedSearchCV",
    )
    parser.add_argument(
        "--tune-n-jobs",
        type=int,
        default=1,
        help="Nombre de jobs paralleles pour RandomizedSearchCV (1 recommande sous Windows)",
    )
    parser.add_argument(
        "--train-n-jobs",
        type=int,
        default=-1,
        help="Nombre de threads XGBoost pour l'entrainement final (-1 = tous les coeurs)",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=50,
        help="Nombre de rounds pour early stopping (0 pour desactiver)",
    )
    parser.add_argument("--max-depth", type=int, default=6, help="Profondeur max des arbres")
    parser.add_argument("--min-child-weight", type=float, default=4.0, help="Poids mini par noeud feuille")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gain mini pour split")
    parser.add_argument("--reg-alpha", type=float, default=0.1, help="Regularisation L1")
    parser.add_argument("--reg-lambda", type=float, default=2.0, help="Regularisation L2")
    parser.add_argument(
        "--disable-phenology",
        action="store_true",
        help="Desactive les features phenologiques (pic, amplitude, AUC, etc.)",
    )
    parser.add_argument(
        "--max-dates",
        type=int,
        default=0,
        help="Nombre max de dates a conserver (0 = toutes)",
    )
    parser.add_argument("--output-dir", default="outputs_xgboost", help="Dossier de sortie")
    return parser.parse_args()


def load_labels(path: Path, id_col: str, target_col: str) -> pd.DataFrame:
    labels = pd.read_csv(path, usecols=[id_col, target_col])
    labels = labels.dropna(subset=[target_col]).copy()
    labels[id_col] = labels[id_col].astype(str)
    labels[target_col] = labels[target_col].astype(str)
    labels = labels.drop_duplicates(subset=[id_col])
    return labels


def aggregate_indices(
    path: Path,
    id_set: set,
    indices_keep: set,
    cloud_max: float,
    px_count_min: int,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    cols = ["date", "ID_PARCEL", "index", "value_mean", "px_count", "cloud_scene"]
    agg_chunks = []

    for chunk in pd.read_csv(path, usecols=cols, chunksize=chunksize):
        chunk = chunk.dropna(subset=["date", "ID_PARCEL", "index", "value_mean", "px_count", "cloud_scene"])
        chunk["ID_PARCEL"] = chunk["ID_PARCEL"].astype(str)
        chunk = chunk[chunk["ID_PARCEL"].isin(id_set)]
        chunk = chunk[chunk["index"].isin(indices_keep)]
        chunk = chunk[chunk["cloud_scene"] <= cloud_max]
        chunk = chunk[chunk["px_count"] >= px_count_min]
        if chunk.empty:
            continue

        chunk["weighted"] = chunk["value_mean"] * chunk["px_count"]
        g = (
            chunk.groupby(["ID_PARCEL", "date", "index"], as_index=False)
            .agg(weighted_sum=("weighted", "sum"), px_sum=("px_count", "sum"))
        )
        agg_chunks.append(g)

    if not agg_chunks:
        raise RuntimeError("Aucune ligne conservee apres filtrage des indices.")

    agg = pd.concat(agg_chunks, ignore_index=True)
    agg = (
        agg.groupby(["ID_PARCEL", "date", "index"], as_index=False)
        .agg(weighted_sum=("weighted_sum", "sum"), px_sum=("px_sum", "sum"))
    )
    agg["value"] = agg["weighted_sum"] / agg["px_sum"]
    return agg[["ID_PARCEL", "date", "index", "value"]]


def limit_dates(agg: pd.DataFrame, max_dates: int) -> pd.DataFrame:
    if max_dates <= 0:
        return agg

    date_counts = agg.groupby("date")["ID_PARCEL"].nunique().sort_values(ascending=False)
    keep_dates = set(date_counts.head(max_dates).index)
    return agg[agg["date"].isin(keep_dates)].copy()


def build_wide_features(agg: pd.DataFrame) -> pd.DataFrame:
    feat_name = agg["index"].astype(str) + "__" + agg["date"].astype(str)
    tmp = agg[["ID_PARCEL", "value"]].copy()
    tmp["feature"] = feat_name

    wide = tmp.pivot_table(index="ID_PARCEL", columns="feature", values="value", aggfunc="mean")
    wide.columns.name = None
    wide = wide.reset_index()
    return wide


def build_phenology_features(agg: pd.DataFrame) -> pd.DataFrame:
    gkeys = ["ID_PARCEL", "index"]
    base = agg[["ID_PARCEL", "index", "date", "value"]].copy()
    base["date_dt"] = pd.to_datetime(base["date"], errors="coerce")
    base = base.dropna(subset=["date_dt"]).copy()
    base["doy"] = base["date_dt"].dt.dayofyear.astype(float)

    stats = (
        base.groupby(gkeys)["value"]
        .agg(["mean", "std", "min", "max", "median"])
        .reset_index()
    )
    stats["amp"] = stats["max"] - stats["min"]

    idx_max = base.groupby(gkeys)["value"].idxmax()
    idx_min = base.groupby(gkeys)["value"].idxmin()
    peak = base.loc[idx_max, gkeys + ["doy"]].rename(columns={"doy": "doy_peak"})
    trough = base.loc[idx_min, gkeys + ["doy"]].rename(columns={"doy": "doy_trough"})

    srt = base.sort_values(gkeys + ["doy"]).copy()
    srt["doy_next"] = srt.groupby(gkeys)["doy"].shift(-1)
    srt["val_next"] = srt.groupby(gkeys)["value"].shift(-1)
    srt["auc_seg"] = 0.5 * (srt["value"] + srt["val_next"]) * (srt["doy_next"] - srt["doy"])
    auc = srt.groupby(gkeys, as_index=False)["auc_seg"].sum().rename(columns={"auc_seg": "auc"})

    pheno = stats.merge(peak, on=gkeys, how="left").merge(trough, on=gkeys, how="left").merge(auc, on=gkeys, how="left")
    pheno_wide = pheno.set_index(gkeys).unstack("index")
    pheno_wide.columns = [f"{idx}__{feat}" for feat, idx in pheno_wide.columns]
    pheno_wide = pheno_wide.reset_index()
    return pheno_wide


def interpolate_by_index(X: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in X.columns if "__" in c]
    temporal_cols = []
    for col in cols:
        _, suffix = col.split("__", 1)
        parsed = pd.to_datetime(suffix, errors="coerce")
        if not pd.isna(parsed):
            temporal_cols.append(col)

    grouped = {}
    for col in temporal_cols:
        idx_name, date_str = col.split("__", 1)
        grouped.setdefault(idx_name, []).append((date_str, col))

    for idx_name, items in grouped.items():
        sorted_cols = [c for _, c in sorted(items, key=lambda x: x[0])]
        X[sorted_cols] = X[sorted_cols].interpolate(axis=1, limit_direction="both")

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    medians = X[numeric_cols].median(axis=0)
    X[numeric_cols] = X[numeric_cols].fillna(medians)
    return X


def augment_train_data(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    method: str,
    random_state: int,
    k_neighbors: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    if method == "none":
        return X_train, y_train

    try:
        if method == "smote":
            from imblearn.over_sampling import SMOTE

            sampler = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        elif method == "borderline_smote":
            from imblearn.over_sampling import BorderlineSMOTE

            sampler = BorderlineSMOTE(random_state=random_state, k_neighbors=k_neighbors)
        else:
            raise ValueError(f"Methode d'augmentation inconnue: {method}")
    except ImportError as exc:
        raise SystemExit(
            "imbalanced-learn n'est pas installe. Lance: pip install imbalanced-learn"
        ) from exc

    X_res, y_res = sampler.fit_resample(X_train, y_train)
    if not isinstance(X_res, pd.DataFrame):
        X_res = pd.DataFrame(X_res, columns=X_train.columns)
    return X_res, y_res


def make_xgb_model(
    n_classes: int,
    random_state: int,
    n_jobs: int,
    max_depth: int,
    min_child_weight: float,
    gamma: float,
    reg_alpha: float,
    reg_lambda: float,
    overrides: dict | None = None,
) -> XGBClassifier:
    params = {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "n_estimators": 600,
        "learning_rate": 0.05,
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": gamma,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "tree_method": "hist",
        "random_state": random_state,
        "n_jobs": n_jobs,
        "eval_metric": "mlogloss",
    }
    if overrides:
        params.update(overrides)
    return XGBClassifier(**params)


def stratified_cv_macro_f1(
    model: XGBClassifier,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    cv_folds: int,
    random_state: int,
) -> tuple[float, float]:
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = []
    for tr_idx, va_idx in skf.split(X_train, y_train):
        fold_model = clone(model)
        X_tr = X_train.iloc[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_tr = y_train[tr_idx]
        y_va = y_train[va_idx]
        w_tr = sample_weight[tr_idx]
        fold_model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
        pred = fold_model.predict(X_va)
        scores.append(f1_score(y_va, pred, average="macro"))
    return float(np.mean(scores)), float(np.std(scores))


def aggregate_feature_importance(model: XGBClassifier, feature_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    booster = model.get_booster()
    raw_gain = booster.get_score(importance_type="gain")

    gain_rows = []
    for k, v in raw_gain.items():
        if k.startswith("f") and k[1:].isdigit():
            idx = int(k[1:])
            if idx < len(feature_names):
                name = feature_names[idx]
            else:
                continue
        else:
            name = k
        gain_rows.append((name, float(v)))

    if not gain_rows:
        empty = pd.DataFrame(columns=["feature", "gain"])
        return empty, pd.DataFrame(columns=["date", "gain"]), pd.DataFrame(columns=["index", "gain"])

    feat_imp = pd.DataFrame(gain_rows, columns=["feature", "gain"]).sort_values("gain", ascending=False)

    tmp = feat_imp.copy()
    tmp[["index", "suffix"]] = tmp["feature"].str.split("__", n=1, expand=True)
    tmp["date"] = pd.to_datetime(tmp["suffix"], errors="coerce")

    date_imp = (
        tmp.dropna(subset=["date"])
        .assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d"))
        .groupby("date", as_index=False)["gain"]
        .sum()
        .sort_values("gain", ascending=False)
    )
    index_imp = tmp.groupby("index", as_index=False)["gain"].sum().sort_values("gain", ascending=False)
    return feat_imp, date_imp, index_imp


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(Path(args.labels_csv), args.id_col, args.target_col)
    id_set = set(labels[args.id_col].tolist())

    print(f"Labels: {len(labels):,} parcelles, {labels[args.target_col].nunique()} classes")

    agg = aggregate_indices(
        path=Path(args.indices_csv),
        id_set=id_set,
        indices_keep=set(args.indices),
        cloud_max=args.cloud_max,
        px_count_min=args.px_count_min,
    )
    agg = limit_dates(agg, args.max_dates)
    print(f"Indices agreges: {len(agg):,} lignes, {agg['date'].nunique()} dates, {agg['index'].nunique()} indices")

    X_wide = build_wide_features(agg).rename(columns={"ID_PARCEL": args.id_col})
    data = labels.merge(X_wide, on=args.id_col, how="inner")

    if not args.disable_phenology:
        pheno_wide = build_phenology_features(agg).rename(columns={"ID_PARCEL": args.id_col})
        data = data.merge(pheno_wide, on=args.id_col, how="left")

    class_counts = data[args.target_col].value_counts()
    keep_classes = class_counts[class_counts >= args.min_class_count].index
    data = data[data[args.target_col].isin(keep_classes)].copy()

    if data.empty:
        raise RuntimeError("Aucune donnee exploitable apres filtrage des classes.")

    y_str = data[args.target_col].astype(str)
    drop_cols = [c for c in [args.id_col, args.target_col] if c in data.columns]
    X = data.drop(columns=drop_cols).copy()

    if X.shape[1] == 0:
        raise RuntimeError("Aucune feature disponible apres preparation des donnees.")

    missing_before = float(X.isna().mean().mean())
    X = interpolate_by_index(X)
    missing_after = float(X.isna().mean().mean())

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    train_size_before_aug = len(y_train)
    X_train, y_train = augment_train_data(
        X_train=X_train,
        y_train=y_train,
        method=args.augment,
        random_state=args.random_state,
        k_neighbors=args.smote_k_neighbors,
    )
    train_size_after_aug = len(y_train)

    class_counts_train = np.bincount(y_train)
    n_classes = len(class_counts_train)
    n_samples = len(y_train)
    class_weights = n_samples / (n_classes * class_counts_train)
    sample_weight = class_weights[y_train]

    best_params = {}
    cv_macro_mean = None
    cv_macro_std = None
    cv_source = "disabled"

    if args.tune_n_iter > 0 and args.cv_folds >= 2:
        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
        tune_model = make_xgb_model(
            n_classes=n_classes,
            random_state=args.random_state,
            n_jobs=1,
            max_depth=args.max_depth,
            min_child_weight=args.min_child_weight,
            gamma=args.gamma,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
        )
        param_dist = {
            "n_estimators": [300, 500, 700, 900],
            "learning_rate": [0.02, 0.03, 0.05, 0.08],
            "max_depth": [4, 6, 8, 10],
            "min_child_weight": [1, 2, 4, 6],
            "subsample": [0.6, 0.7, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
            "gamma": [0.0, 0.1, 0.2, 0.5],
            "reg_alpha": [0.0, 0.1, 0.5, 1.0],
            "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        }
        search = RandomizedSearchCV(
            estimator=tune_model,
            param_distributions=param_dist,
            n_iter=args.tune_n_iter,
            scoring="f1_macro",
            n_jobs=args.tune_n_jobs,
            cv=cv,
            random_state=args.random_state,
            verbose=args.tune_verbose,
            refit=True,
            pre_dispatch=max(1, args.tune_n_jobs) if args.tune_n_jobs != -1 else "2*n_jobs",
        )
        try:
            search.fit(X_train, y_train, sample_weight=sample_weight)
        except OSError as exc:
            if "WinError 1450" in str(exc) and args.tune_n_jobs != 1:
                print("WinError 1450 detecte pendant le tuning, retry automatique avec --tune-n-jobs=1")
                search.set_params(n_jobs=1, pre_dispatch=1)
                search.fit(X_train, y_train, sample_weight=sample_weight)
            else:
                raise
        best_params = search.best_params_
        cv_macro_mean = float(search.best_score_)
        cv_macro_std = float(search.cv_results_["std_test_score"][search.best_index_])
        cv_source = "randomized_search_cv"
        print(f"Tuning termine | Best CV Macro-F1: {cv_macro_mean:.4f} (+/- {cv_macro_std:.4f})")
    elif args.cv_folds >= 2:
        cv_source = "stratified_kfold_baseline"

    model = make_xgb_model(
        n_classes=n_classes,
        random_state=args.random_state,
        n_jobs=args.train_n_jobs,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        overrides=best_params,
    )

    if cv_source == "stratified_kfold_baseline":
        cv_macro_mean, cv_macro_std = stratified_cv_macro_f1(
            model=model,
            X_train=X_train,
            y_train=y_train,
            sample_weight=sample_weight,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
        )
        print(f"CV Macro-F1 (baseline): {cv_macro_mean:.4f} (+/- {cv_macro_std:.4f})")

    X_fit, X_val, y_fit, y_val, w_fit, _ = train_test_split(
        X_train,
        y_train,
        sample_weight,
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=y_train,
    )

    fit_kwargs = {
        "sample_weight": w_fit,
        "eval_set": [(X_val, y_val)],
        "verbose": False,
    }
    early_stopping_mode = "disabled"
    if args.early_stopping_rounds > 0:
        try:
            # Compatible avec les versions recentes: parametre sur l'estimateur.
            model.set_params(early_stopping_rounds=args.early_stopping_rounds)
            early_stopping_mode = "estimator_param"
        except ValueError:
            try:
                # Fallback pour certaines versions qui passent par callback.
                import xgboost as xgb

                fit_kwargs["callbacks"] = [
                    xgb.callback.EarlyStopping(rounds=args.early_stopping_rounds, save_best=True)
                ]
                early_stopping_mode = "callback"
            except Exception:
                early_stopping_mode = "unsupported"

    model.fit(X_fit, y_fit, **fit_kwargs)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_accuracy = balanced_accuracy_score(y_test, y_pred)
    macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    class_names = encoder.classes_
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_row_sum = cm_df.sum(axis=1).replace(0, np.nan)
    df_norm = cm_df.div(cm_row_sum, axis=0).fillna(0.0)

    feature_names = X.columns.tolist()
    feat_imp, date_imp, index_imp = aggregate_feature_importance(model, feature_names)

    metrics = {
        "n_samples": int(len(data)),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(class_names)),
        "augmentation": args.augment,
        "phenology_enabled": bool(not args.disable_phenology),
        "train_size_before_augmentation": int(train_size_before_aug),
        "train_size_after_augmentation": int(train_size_after_aug),
        "cv_folds": int(args.cv_folds),
        "cv_source": cv_source,
        "val_size": float(args.val_size),
        "early_stopping_rounds": int(args.early_stopping_rounds),
        "early_stopping_mode": early_stopping_mode,
        "cv_macro_f1_mean": None if cv_macro_mean is None else float(cv_macro_mean),
        "cv_macro_f1_std": None if cv_macro_std is None else float(cv_macro_std),
        "best_params": best_params,
        "missing_before_imputation": missing_before,
        "missing_after_imputation": missing_after,
        "accuracy": float(acc),
        "macro_accuracy": float(macro_accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_df.to_csv(out_dir / "classification_report.csv", index=True)
    cm_df.to_csv(out_dir / "confusion_matrix.csv", index=True)
    df_norm.to_csv(out_dir / "confusion_matrix_normalized.csv", index=True)
    feat_imp.to_csv(out_dir / "feature_importance_gain.csv", index=False)
    date_imp.to_csv(out_dir / "date_importance_gain.csv", index=False)
    index_imp.to_csv(out_dir / "index_importance_gain.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.imshow(df_norm, cmap="Greens")
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=90)
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Classe prédite")
    plt.ylabel("Classe réelle")
    plt.title("Matrice de confusion (normalisée)")
    ax = plt.gca()
    ax.set_frame_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix_normalized.png", dpi=200)
    plt.close()

    print("\n=== Resultats XGBoost ===")
    print(f"Samples: {metrics['n_samples']:,} | Features: {metrics['n_features']:,} | Classes: {metrics['n_classes']}")
    print(f"Augmentation: {metrics['augmentation']} | Train avant/apres: {metrics['train_size_before_augmentation']:,}/{metrics['train_size_after_augmentation']:,}")
    print(f"Phenology: {metrics['phenology_enabled']}")
    if metrics["cv_macro_f1_mean"] is not None:
        print(f"CV Macro-F1: {metrics['cv_macro_f1_mean']:.4f} (+/- {metrics['cv_macro_f1_std']:.4f}) [{metrics['cv_source']}]")
    print(f"Missing (avant): {metrics['missing_before_imputation']:.4f}")
    print(f"Missing (apres): {metrics['missing_after_imputation']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-Accuracy: {metrics['macro_accuracy']:.4f}")
    print(f"Macro-Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro-Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted-F1: {metrics['weighted_f1']:.4f}")
    print(f"\nFichiers ecrits dans: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
