"""Stage 1: fit Isolation Forest, pseudo-label train, persist train/test split."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

from pipeline_common import (
    DEFAULT_ARTIFACTS,
    DEFAULT_CSV,
    TARGET,
    load_data,
    save_stage1_artifacts,
    stage1_dir,
)


def run(args) -> Path:
    df, features = load_data(Path(args.csv), args.max_rows)
    if len(df) < 150:
        raise ValueError('Need at least 150 rows after sampling; raise --max-rows')
    train_end = int(len(df) * args.train_fraction)
    train = df.iloc[:train_end].copy()
    test = df.iloc[train_end:].copy()

    imputer = SimpleImputer(strategy='median')
    x_train = imputer.fit_transform(train[features])
    if_model = IsolationForest(
        n_estimators=150, contamination=args.contamination, random_state=42,
    )
    if_model.fit(x_train)
    training_scores = if_model.score_samples(x_train)
    threshold = float(np.quantile(training_scores, args.contamination))
    train[TARGET] = (training_scores < threshold).astype(int)
    n_anom = int(train[TARGET].sum())
    print(f'Train: {len(train)} rows | IF pseudo-labels {n_anom}/{len(train)} '
          f'anomalies (threshold={threshold:.4f})')
    print(f'Test: {len(test)} rows (held out; not used for fitting)')

    out = stage1_dir(Path(args.artifacts))
    meta = {
        'features': features,
        'train_fraction': args.train_fraction,
        'contamination': args.contamination,
        'threshold': threshold,
        'max_rows': args.max_rows,
        'csv': str(Path(args.csv).resolve()),
        'n_train': len(train),
        'n_test': len(test),
        'n_train_anomalies': n_anom,
    }
    # Keep timestamp + features + label on train; test has timestamp + features.
    train_cols = ['timestamp'] + features + [TARGET]
    test_cols = ['timestamp'] + features
    save_stage1_artifacts(
        out,
        meta=meta,
        imputer=imputer,
        if_model=if_model,
        train=train[train_cols],
        test=test[test_cols],
    )
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=DEFAULT_CSV)
    parser.add_argument('--max-rows', type=float, default=1.0,
                        help='Percent of CSV (chronological prefix)')
    parser.add_argument('--train-fraction', type=float, default=0.6)
    parser.add_argument('--contamination', type=float, default=0.02)
    parser.add_argument('--artifacts', type=Path, default=DEFAULT_ARTIFACTS,
                        help='Root artifacts directory (writes artifacts/stage1)')
    args = parser.parse_args()
    if not args.csv.exists():
        parser.error(f'CSV not found: {args.csv}')
    if not 0 < args.max_rows <= 100:
        parser.error('--max-rows must be in (0, 100]')
    if not 0 < args.train_fraction < 1 or not 0 < args.contamination < 0.5:
        parser.error('Invalid train-fraction or contamination')
    run(args)
