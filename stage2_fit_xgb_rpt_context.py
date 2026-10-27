"""Stage 2: train XGBoost on IF pseudo-labels; build balanced RPT few-shot context."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pipeline_common import (
    DEFAULT_ARTIFACTS,
    TARGET,
    build_rpt_context,
    fit_xgboost,
    load_stage1_artifacts,
    report_xgb_importance,
    save_stage2_artifacts,
    select_rpt_features,
    stage1_dir,
    stage2_dir,
)


def run(args) -> Path:
    s1 = load_stage1_artifacts(stage1_dir(Path(args.artifacts)))
    train = s1['train']
    features = s1['features']
    imputer = s1['imputer']
    if_model = s1['if_model']

    x_train = imputer.transform(train[features])
    y_train = train[TARGET].to_numpy()
    xgb = fit_xgboost(x_train, y_train)

    out = stage2_dir(Path(args.artifacts))
    out.mkdir(parents=True, exist_ok=True)
    importance_path = out / 'xgb_feature_importance.csv'
    ranked = report_xgb_importance(xgb, features, importance_path)

    rpt_features = select_rpt_features(ranked, args.rpt_top_features)
    print(f'RPT features ({len(rpt_features)}): {rpt_features}')

    if_scores = None
    if args.context_selection == 'boundary':
        if_scores = np.asarray(if_model.score_samples(x_train), dtype=float)
    context = build_rpt_context(
        train,
        rpt_features,
        args.context_size,
        neg_frac=args.context_neg_frac,
        selection=args.context_selection,
        if_scores=if_scores,
    )
    # Persist only columns RPT needs.
    ctx_cols = [c for c in (['timestamp'] + rpt_features + [TARGET]) if c in context.columns]
    context = context[ctx_cols]

    meta = {
        'context_size': args.context_size,
        'context_neg_frac': args.context_neg_frac,
        'context_selection': args.context_selection,
        'n_context': len(context),
        'n_context_pos': int(context[TARGET].sum()),
        'features': features,
        'rpt_features': rpt_features,
        'rpt_top_features': args.rpt_top_features,
        'stage1_dir': str(stage1_dir(Path(args.artifacts)).resolve()),
    }
    save_stage2_artifacts(
        out,
        meta=meta,
        xgb=xgb,
        context=context,
        importance_path=importance_path,
    )
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=DEFAULT_ARTIFACTS,
                        help='Root artifacts directory (reads stage1, writes stage2)')
    parser.add_argument('--context-size', type=int, default=64,
                        help='RPT few-shot context size (try 96-128)')
    parser.add_argument('--context-neg-frac', type=float, default=0.5,
                        help='Fraction of context rows that are IF-normal')
    parser.add_argument(
        '--context-selection', choices=('spaced', 'boundary'), default='spaced',
        help="spaced=time-spread (default); boundary=near-IF-threshold (experimental)",
    )
    parser.add_argument('--rpt-top-features', type=int, default=0,
                        help='Use top-N XGB-gain features for RPT (0 = all sensors)')
    args = parser.parse_args()
    s1_path = stage1_dir(args.artifacts)
    if not (s1_path / 'meta.json').exists():
        parser.error(f'Stage1 artifacts not found in {s1_path}; run stage1_fit_if.py first')
    if args.context_size < 2:
        parser.error('--context-size must be >= 2')
    if not 0.1 <= args.context_neg_frac <= 0.9:
        parser.error('--context-neg-frac must be in [0.1, 0.9]')
    run(args)
