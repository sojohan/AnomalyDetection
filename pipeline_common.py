"""Shared helpers for the MetroPT three-script IF / XGB / RPT pipeline."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

META = {'Unnamed: 0', 'timestamp'}
TARGET = 'is_anomaly'
REQUIRED = [
    'TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Motor_current',
]
DEFAULT_CSV = Path(__file__).resolve().parent / 'MetroPT3(AirCompressor).csv'
DEFAULT_ARTIFACTS = Path(__file__).resolve().parent / 'artifacts'
STAGE1_DIRNAME = 'stage1'
STAGE2_DIRNAME = 'stage2'


def stage1_dir(artifacts: Path) -> Path:
    return Path(artifacts) / STAGE1_DIRNAME


def stage2_dir(artifacts: Path) -> Path:
    return Path(artifacts) / STAGE2_DIRNAME


def load_data(path: Path, sample_pct: float) -> tuple[pd.DataFrame, list[str]]:
    """Load CSV, keep chronological prefix, return frame + feature column names."""
    frame = pd.read_csv(path, sep=None, engine='python')
    frame.columns = frame.columns.str.strip()
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f'Missing columns: {missing}; available: {list(frame.columns)}')
    if not 0 < sample_pct <= 100:
        raise ValueError('--max-rows must be a percentage in (0, 100]')
    features = sorted(c for c in frame.columns if c not in META)
    if not features:
        raise ValueError('No sensor feature columns found')
    n_total = len(frame)
    n_keep = max(1, int(round(n_total * sample_pct / 100.0)))
    frame = frame.iloc[:n_keep].copy()
    for col in features:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    if 'timestamp' not in frame:
        frame['timestamp'] = np.arange(len(frame)).astype(str)
    print(f'Loaded {path.name}: {len(frame):,} of {n_total:,} rows '
          f'({sample_pct:g}% chronological prefix)')
    print(f'Features ({len(features)}): {features}')
    return frame, features


def parse_anomaly_label(value) -> int:
    """Map RPT classification output to 0/1."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(float(value) >= 0.5)
    text = str(value).strip().lower()
    if text in ('1', '1.0', 'true', 'anomaly', 'anomalous', 'yes'):
        return 1
    if text in ('0', '0.0', 'false', 'normal', 'no'):
        return 0
    try:
        return int(float(text) >= 0.5)
    except ValueError as exc:
        raise ValueError(f'Cannot parse anomaly label: {value!r}') from exc


class RPTPredictor:
    def __init__(self, context: pd.DataFrame, features: list[str], mock: bool,
                 threshold: float, imputer: SimpleImputer, if_model: IsolationForest):
        self.mock = mock
        self.features = features
        self.threshold = threshold
        self.imputer = imputer
        self.if_model = if_model
        cols = features + [TARGET]
        self.context = context[cols].dropna().to_dict('records')
        if not self.context:
            raise ValueError('No complete rows available for RPT context')
        if not mock:
            from gen_ai_hub.proxy.native.sap import RPTClient
            self.client = RPTClient()

    def predict(self, features: dict) -> int:
        if self.mock:
            vector = self.imputer.transform(
                pd.DataFrame([features], columns=self.features)
            )
            score = float(self.if_model.score_samples(vector)[0])
            return int(score < self.threshold)
        from gen_ai_hub.proxy.native.sap import PredictionConfig, RPTRequest, TargetColumn
        rows = self.context + [{**features, TARGET: '[PREDICT]'}]
        body = RPTRequest(
            prediction_config=PredictionConfig(
                target_columns=[TargetColumn(name=TARGET, task_type='classification')]
            ),
            rows=rows,
        )
        result = self.client.predict(body=body, model_name='sap-rpt-1.6')
        prediction = result.predictions[-1]
        root = getattr(prediction, 'root', prediction)
        if hasattr(root, 'model_dump'):
            root = root.model_dump()
        items = root[TARGET]
        item = items[0] if isinstance(items, list) else items
        value = getattr(item, 'prediction', None)
        if value is None and isinstance(item, dict):
            value = item.get('prediction')
        if value is None:
            raise ValueError(f'Unexpected RPT response shape: {result!r}')
        return parse_anomaly_label(value)


def fit_xgboost(x_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    """Train a binary classifier on Isolation Forest pseudo-labels."""
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f'XGBoost needs both classes in train labels; got pos={n_pos} neg={n_neg}. '
            'Raise --max-rows or --contamination.'
        )
    clf = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=n_neg / n_pos,
        random_state=42,
        n_jobs=1,
    )
    clf.fit(x_train, y_train)
    return clf


def report_xgb_importance(
    clf: XGBClassifier, features: list[str], path: Path,
) -> list[tuple[str, float]]:
    """Write and print XGBoost gain importance for stage-2 features."""
    try:
        booster = clf.get_booster()
        raw = booster.get_score(importance_type='gain')
        scores = np.zeros(len(features), dtype=float)
        for key, value in raw.items():
            idx = int(key[1:]) if key.startswith('f') else features.index(key)
            scores[idx] = float(value)
    except Exception:
        scores = np.asarray(clf.feature_importances_, dtype=float)
    total = float(scores.sum())
    if total > 0:
        scores = scores / total
    ranked = sorted(zip(features, scores), key=lambda x: x[1], reverse=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['feature', 'importance_gain'])
        writer.writeheader()
        for name, score in ranked:
            writer.writerow({'feature': name, 'importance_gain': round(float(score), 6)})
    print(f'Stage2 XGB feature importance (gain, normalized) -> {path}')
    for name, score in ranked:
        print(f'  {name}: {score:.4f}')
    return ranked


def should_run_agent(trigger: str, xgb_pred: int, rpt_pred: int) -> bool:
    """Decide whether a stage-2 event warrants an LLM investigation."""
    if trigger == 'confirmed':
        return xgb_pred == 1 and rpt_pred == 1
    if trigger == 'all':
        return True
    raise ValueError(f'Unknown agent trigger: {trigger!r}')


def _spaced_indices(n: int, k: int) -> np.ndarray:
    """Pick up to k equally spaced indices from a length-n sequence."""
    if n <= 0 or k <= 0:
        return np.array([], dtype=int)
    k = min(k, n)
    if k == 1:
        return np.array([n // 2], dtype=int)
    return np.linspace(0, n - 1, k, dtype=int)


def select_rpt_features(
    ranked_importance: list[tuple[str, float]], top_k: int,
) -> list[str]:
    """Keep the top-k XGB-gain features for RPT (0 / negative = all)."""
    names = [name for name, _ in ranked_importance]
    if top_k is None or top_k <= 0 or top_k >= len(names):
        return names
    return names[:top_k]


def build_rpt_context(
    train: pd.DataFrame,
    features: list[str],
    size: int,
    *,
    neg_frac: float = 0.5,
    selection: str = 'spaced',
    if_scores: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build class-balanced few-shot RPT context.

    selection='spaced' (default): equally spaced pos/neg — known-good baseline.
    selection='boundary': strongest IF anomalies + normals nearest the threshold
    (experimental; can hurt live RPT if boundary normals look like alerts).
    """
    mask = train[features].notna().all(axis=1)
    usable = train.loc[mask].copy()
    use_boundary = selection == 'boundary'
    if use_boundary:
        if if_scores is None:
            raise ValueError("selection='boundary' requires if_scores")
        scores = np.asarray(if_scores, dtype=float)
        if len(scores) != len(train):
            raise ValueError('if_scores length must match train rows')
        usable['_if_score'] = scores[mask.to_numpy()]

    pos = usable[usable[TARGET] == 1]
    neg = usable[usable[TARGET] == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError(
            'RPT context needs both IF classes in train; raise --max-rows or --contamination'
        )
    neg_frac = min(max(float(neg_frac), 0.1), 0.9)
    n_neg = min(len(neg), max(1, int(round(size * neg_frac))))
    n_pos = min(len(pos), max(1, size - n_neg))
    if n_pos + n_neg < size:
        n_neg = min(len(neg), size - n_pos)

    if use_boundary:
        pos_rows = pos.nsmallest(n_pos, '_if_score').drop(columns='_if_score')
        neg_rows = neg.nsmallest(n_neg, '_if_score').drop(columns='_if_score')
    else:
        pos_rows = pos.iloc[_spaced_indices(len(pos), n_pos)]
        neg_rows = neg.iloc[_spaced_indices(len(neg), n_neg)]

    context = pd.concat([pos_rows, neg_rows], ignore_index=True)
    if 'timestamp' in context.columns:
        context = context.sort_values('timestamp').reset_index(drop=True)
    print(
        f'RPT context: {len(context)} rows '
        f'(pos={int(context[TARGET].sum())} neg={int((context[TARGET] == 0).sum())}) '
        f'| neg_frac={neg_frac:.2f} selection={selection} '
        f'| train pool pos={len(pos)} neg={len(neg)}'
    )
    return context


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def save_joblib(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path: Path) -> Any:
    return joblib.load(path)


def save_stage1_artifacts(
    out_dir: Path,
    *,
    meta: dict[str, Any],
    imputer: SimpleImputer,
    if_model: IsolationForest,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / 'meta.json', meta)
    save_joblib(out_dir / 'imputer.joblib', imputer)
    save_joblib(out_dir / 'isolation_forest.joblib', if_model)
    train.to_csv(out_dir / 'train_labeled.csv', index=False)
    test.to_csv(out_dir / 'test.csv', index=False)
    print(f'Stage1 artifacts -> {out_dir}')


def load_stage1_artifacts(out_dir: Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    meta = read_json(out_dir / 'meta.json')
    train = pd.read_csv(out_dir / 'train_labeled.csv')
    test = pd.read_csv(out_dir / 'test.csv')
    features = list(meta['features'])
    for col in features:
        if col in train.columns:
            train[col] = pd.to_numeric(train[col], errors='coerce')
        if col in test.columns:
            test[col] = pd.to_numeric(test[col], errors='coerce')
    if TARGET in train.columns:
        train[TARGET] = pd.to_numeric(train[TARGET], errors='coerce').astype(int)
    return {
        'meta': meta,
        'imputer': load_joblib(out_dir / 'imputer.joblib'),
        'if_model': load_joblib(out_dir / 'isolation_forest.joblib'),
        'train': train,
        'test': test,
        'features': features,
        'threshold': float(meta['threshold']),
    }


def save_stage2_artifacts(
    out_dir: Path,
    *,
    meta: dict[str, Any],
    xgb: XGBClassifier,
    context: pd.DataFrame,
    importance_path: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / 'meta.json', meta)
    save_joblib(out_dir / 'xgboost.joblib', xgb)
    context.to_csv(out_dir / 'rpt_context.csv', index=False)
    # importance already written to importance_path; copy name into stage2 if needed
    if importance_path.resolve() != (out_dir / 'xgb_feature_importance.csv').resolve():
        import shutil
        shutil.copy2(importance_path, out_dir / 'xgb_feature_importance.csv')
    print(f'Stage2 artifacts -> {out_dir}')


def load_stage2_artifacts(out_dir: Path, features: list[str]) -> dict[str, Any]:
    out_dir = Path(out_dir)
    meta = read_json(out_dir / 'meta.json')
    context = pd.read_csv(out_dir / 'rpt_context.csv')
    rpt_features = list(meta.get('rpt_features') or features)
    for col in rpt_features + [TARGET]:
        if col in context.columns:
            context[col] = pd.to_numeric(context[col], errors='coerce')
    if TARGET in context.columns:
        context[TARGET] = context[TARGET].astype(int)
    return {
        'meta': meta,
        'xgb': load_joblib(out_dir / 'xgboost.joblib'),
        'context': context,
    }
