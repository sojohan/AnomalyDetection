"""Orchestrate the three-script MetroPT pipeline (IF -> XGB/RPT context -> test).

Runs stage1_fit_if, stage2_fit_xgb_rpt_context, then stage3_score_test in order.
You can also run those scripts individually; see README.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import stage1_fit_if
import stage2_fit_xgb_rpt_context
import stage3_score_test
from pipeline_common import DEFAULT_ARTIFACTS, DEFAULT_CSV


def run(args) -> None:
    artifacts = Path(args.artifacts)
    print('=== Stage 1: fit Isolation Forest ===')
    stage1_fit_if.run(SimpleNamespace(
        csv=args.csv,
        max_rows=args.max_rows,
        train_fraction=args.train_fraction,
        contamination=args.contamination,
        artifacts=artifacts,
    ))
    print('=== Stage 2: train XGBoost + RPT context ===')
    stage2_fit_xgb_rpt_context.run(SimpleNamespace(
        artifacts=artifacts,
        context_size=args.context_size,
        context_neg_frac=args.context_neg_frac,
        context_selection=args.context_selection,
        rpt_top_features=args.rpt_top_features,
    ))
    print('=== Stage 3: score test (IF gate + XGB/RPT) ===')
    stage3_score_test.run(SimpleNamespace(
        artifacts=artifacts,
        cooldown=args.cooldown,
        sleep=args.sleep,
        print_every=args.print_every,
        mock_rpt=args.mock_rpt,
        output=args.output,
        agents=args.agents,
        agent_model=args.agent_model,
        max_agent_reports=args.max_agent_reports,
        agent_trigger=args.agent_trigger,
        agent_output=args.agent_output,
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=DEFAULT_CSV)
    parser.add_argument('--max-rows', type=float, default=1.0,
                        help='Percent of the CSV to use (chronological prefix)')
    parser.add_argument('--train-fraction', type=float, default=0.6)
    parser.add_argument('--contamination', type=float, default=0.02)
    parser.add_argument('--context-size', type=int, default=64)
    parser.add_argument('--context-neg-frac', type=float, default=0.5,
                        help='Share of RPT context rows that are IF-normal')
    parser.add_argument('--context-selection', choices=('spaced', 'boundary'),
                        default='spaced')
    parser.add_argument('--rpt-top-features', type=int, default=0,
                        help='Top-N XGB features for RPT (0 = all)')
    parser.add_argument('--cooldown', type=int, default=30)
    parser.add_argument('--sleep', type=float, default=0)
    parser.add_argument('--print-every', type=int, default=100)
    parser.add_argument('--mock-rpt', action='store_true')
    parser.add_argument('--output', default='results/stream_predictions.csv')
    parser.add_argument('--artifacts', type=Path, default=DEFAULT_ARTIFACTS,
                        help='Root directory for stage1/stage2 artifacts')
    parser.add_argument('--agents', action='store_true')
    parser.add_argument('--agent-model', default='openai:gpt-4.1-mini')
    parser.add_argument('--max-agent-reports', type=int, default=3)
    parser.add_argument('--agent-trigger', choices=('confirmed', 'all'), default='confirmed')
    parser.add_argument('--agent-output', default='results/agent_reports.jsonl')
    args = parser.parse_args()
    if not args.csv.exists():
        parser.error(f'CSV not found: {args.csv}')
    if not 0 < args.max_rows <= 100:
        parser.error('--max-rows must be a percentage in (0, 100]')
    if not 0 < args.train_fraction < 1 or not 0 < args.contamination < 0.5:
        parser.error('train-fraction must be between 0 and 1; contamination between 0 and .5')
    if args.max_agent_reports < 1:
        parser.error('--max-agent-reports must be >= 1')
    run(args)
