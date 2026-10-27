"""Evaluate stage-2 models on the held-out test set: accuracy + confusion matrix.

Reference labels are Isolation Forest alerts on test (IF pseudo-labels), not
official MetroPT failure events. XGBoost is scored on every test row. RPT is
scored on every test row by default (use --max-rpt-rows to cap live SAP cost).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from pipeline_common import DEFAULT_ARTIFACTS, stage1_dir, stage2_dir
from scoring import ScoringSession


def _print_block(title: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = [0, 1]
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()
    print(f'\n=== {title} ===')
    print(f'n={len(y_true)} | positives(IF=1)={int(y_true.sum())} | '
          f'pred_pos={int(y_pred.sum())}')
    print(f'Accuracy: {acc:.4f} ({acc:.1%})')
    print('Confusion matrix (rows=IF true 0/1, cols=pred 0/1):')
    print('          pred_0  pred_1')
    print(f'  IF_0  {tn:7d} {fp:7d}')
    print(f'  IF_1  {fn:7d} {tp:7d}')
    print(classification_report(
        y_true, y_pred, labels=labels, target_names=['normal(0)', 'anomaly(1)'],
        digits=4, zero_division=0,
    ))
    return {
        'n': int(len(y_true)),
        'accuracy': acc,
        'confusion_matrix': {
            'labels': [0, 1],
            'matrix': cm.tolist(),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        },
    }


def _rpt_eval_indices(
    if_labels: list[int], max_rpt_rows: int | None,
) -> set[int]:
    """Pick RPT rows: all IF alerts first, then spaced normals to fill the budget."""
    n = len(if_labels)
    if max_rpt_rows is None or max_rpt_rows <= 0 or max_rpt_rows >= n:
        return set(range(n))
    pos = [i for i, y in enumerate(if_labels) if y == 1]
    neg = [i for i, y in enumerate(if_labels) if y == 0]
    chosen = pos[:max_rpt_rows]
    remaining = max_rpt_rows - len(chosen)
    if remaining > 0 and neg:
        step = max(1, len(neg) // remaining)
        chosen.extend(neg[::step][:remaining])
    return set(chosen[:max_rpt_rows])


def run(args) -> Path:
    session = ScoringSession.load(
        args.artifacts, mock_rpt=args.mock_rpt, cooldown=1,
    )
    test = session.test
    features = session.features
    print(
        f'Test rows={len(test)} | IF threshold={session.threshold:.4f} | '
        f'RPT={"MOCK" if args.mock_rpt else "LIVE"} | '
        f'max_rpt_rows={args.max_rpt_rows or "all"}'
    )

    # Pass 1: IF + XGB on all rows (cheap)
    y_if: list[int] = []
    y_xgb: list[int] = []
    rows_cache: list[pd.Series] = []
    for i, (_, row) in enumerate(test.iterrows()):
        vector = session.imputer.transform(
            pd.DataFrame([row[features]], columns=features)
        )
        score = float(session.if_model.score_samples(vector)[0])
        if_label = int(score < session.threshold)
        xgb_pred = int(session.xgb.predict(vector)[0])
        y_if.append(if_label)
        y_xgb.append(xgb_pred)
        rows_cache.append(row)
        if (i + 1) % args.print_every == 0:
            print(f'... IF+XGB {i + 1}/{len(test)}')

    rpt_limit = args.max_rpt_rows if args.max_rpt_rows and args.max_rpt_rows > 0 else None
    rpt_idx = _rpt_eval_indices(y_if, rpt_limit)
    print(f'RPT eval subset: {len(rpt_idx)} rows '
          f'(IF alerts in subset={sum(1 for i in rpt_idx if y_if[i] == 1)})')

    y_if_rpt: list[int] = []
    y_xgb_rpt: list[int] = []
    y_rpt: list[int] = []
    for n_done, i in enumerate(sorted(rpt_idx), start=1):
        row = rows_cache[i]
        rpt_feats = session.rpt.features
        if not row[rpt_feats].notna().all():
            continue
        try:
            feat = {col: float(row[col]) for col in rpt_feats}
            rpt_pred = int(session.rpt.predict(feat))
            y_rpt.append(rpt_pred)
            y_if_rpt.append(y_if[i])
            y_xgb_rpt.append(y_xgb[i])
        except Exception as exc:
            print(f'RPT error at row {i}: {type(exc).__name__}: {exc}')
        if n_done % max(1, args.print_every // 5) == 0:
            print(f'... RPT {n_done}/{len(rpt_idx)}')

    y_if_a = np.asarray(y_if, dtype=int)
    y_xgb_a = np.asarray(y_xgb, dtype=int)
    results: dict = {
        'reference': 'isolation_forest_alert_on_test',
        'caveat': 'IF pseudo-labels, not official MetroPT failure events',
        'threshold': session.threshold,
        'mock_rpt': args.mock_rpt,
        'xgboost': _print_block('XGBoost vs IF (all test rows)', y_if_a, y_xgb_a),
    }

    if y_rpt:
        y_rpt_a = np.asarray(y_rpt, dtype=int)
        y_if_rpt_a = np.asarray(y_if_rpt, dtype=int)
        y_xgb_rpt_a = np.asarray(y_xgb_rpt, dtype=int)
        n_shared = len(y_rpt)
        # Head-to-head: XGB and RPT on the exact same rows
        results['xgboost_on_rpt_rows'] = _print_block(
            f'XGBoost vs IF (same {n_shared} rows as RPT)',
            y_if_rpt_a,
            y_xgb_rpt_a,
        )
        results['rpt'] = _print_block(
            f'RPT vs IF (same {n_shared} rows'
            + ('' if rpt_limit is None else f', capped by --max-rpt-rows {rpt_limit}')
            + ')',
            y_if_rpt_a,
            y_rpt_a,
        )
        agree = float(accuracy_score(y_xgb_rpt_a, y_rpt_a))
        print(f'\n=== XGBoost vs RPT (same {n_shared} rows) ===')
        print(f'Agreement: {agree:.4f} ({agree:.1%})')
        results['xgb_vs_rpt_agreement'] = agree
        # Persist row-level preds for reproducible head-to-head
        rows_path = Path(args.output).with_name('stage2_metrics_rpt_rows.csv')
        pd.DataFrame({
            'if_label': y_if_rpt_a,
            'xgb_pred': y_xgb_rpt_a,
            'rpt_pred': y_rpt_a,
        }).to_csv(rows_path, index=False)
        print(f'Shared-row preds -> {rows_path}')
        results['shared_rows_csv'] = str(rows_path)
    else:
        print('\nNo RPT predictions collected.')
        results['rpt'] = None
        results['xgboost_on_rpt_rows'] = None

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'\nMetrics JSON -> {out}')
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument('--mock-rpt', action='store_true',
                        help='Use mock RPT (echoes IF); for pipeline checks only')
    parser.add_argument('--max-rpt-rows', type=int, default=0,
                        help='Cap RPT calls (0 = all test rows). Recommended for live RPT.')
    parser.add_argument('--print-every', type=int, default=500)
    parser.add_argument('--output', default='results/stage2_metrics.json')
    args = parser.parse_args()
    if not (stage1_dir(args.artifacts) / 'meta.json').exists():
        parser.error('Stage1 artifacts missing; run stage1_fit_if.py first')
    if not (stage2_dir(args.artifacts) / 'meta.json').exists():
        parser.error('Stage2 artifacts missing; run stage2_fit_xgb_rpt_context.py first')
    run(args)
