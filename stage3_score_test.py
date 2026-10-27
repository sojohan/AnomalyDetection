"""Stage 3: score test with IF gate, then XGBoost + RPT (optional Deep Agents)."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from pipeline_common import (
    DEFAULT_ARTIFACTS,
    should_run_agent,
    stage1_dir,
    stage2_dir,
)
from scoring import ScoringSession, build_agent_event


def run(args) -> Path:
    session = ScoringSession.load(
        args.artifacts, mock_rpt=args.mock_rpt, cooldown=args.cooldown,
    )
    test = session.test

    investigator = None
    agent_reports_path = Path(args.agent_output)
    if args.agents:
        from agent_investigator import AnomalyInvestigator
        agent_reports_path.parent.mkdir(parents=True, exist_ok=True)
        agent_reports_path.write_text('', encoding='utf-8')
        investigator = AnomalyInvestigator(model=args.agent_model, output=str(agent_reports_path))
        print(f'Agents ON | model={args.agent_model} | trigger={args.agent_trigger} | '
              f'max_reports={args.max_agent_reports} | -> {agent_reports_path}')

    print(f'Test: {len(test)} rows | stage1=IF every row | '
          f'stage2=XGB+RPT on alerts (cooldown={args.cooldown}) | '
          f'RPT={"MOCK" if args.mock_rpt else "LIVE"}')

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'timestamp', 'actual_current', 'isolation_score', 'isolation_alert',
        'xgb_is_anomaly', 'xgb_agrees_with_if',
        'rpt_is_anomaly', 'rpt_agrees_with_if', 'rpt_agrees_with_xgb', 'rpt_status',
        'agent_status',
    ]
    n_stage1_alerts = 0
    n_stage2 = 0
    n_xgb_if = 0
    n_rpt_if = 0
    n_rpt_xgb = 0
    n_agent_reports = 0

    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, (_, row) in enumerate(test.iterrows()):
            scored = session.score_row(row, index=i)
            if scored['isolation_alert']:
                n_stage1_alerts += 1
            record = {
                'timestamp': scored['timestamp'],
                'actual_current': scored['actual_current'],
                'isolation_score': scored['isolation_score'],
                'isolation_alert': scored['isolation_alert'],
                'xgb_is_anomaly': '' if scored['xgb_is_anomaly'] is None else scored['xgb_is_anomaly'],
                'xgb_agrees_with_if': '' if scored['xgb_agrees_with_if'] is None else scored['xgb_agrees_with_if'],
                'rpt_is_anomaly': '' if scored['rpt_is_anomaly'] is None else scored['rpt_is_anomaly'],
                'rpt_agrees_with_if': '' if scored['rpt_agrees_with_if'] is None else scored['rpt_agrees_with_if'],
                'rpt_agrees_with_xgb': '' if scored['rpt_agrees_with_xgb'] is None else scored['rpt_agrees_with_xgb'],
                'rpt_status': scored['rpt_status'],
                'agent_status': 'not_called',
            }
            if scored['stage2_called'] and scored['rpt_status'] in ('mock', 'live'):
                n_stage2 += 1
                n_xgb_if += int(scored['xgb_agrees_with_if'] or 0)
                n_rpt_if += int(scored['rpt_agrees_with_if'] or 0)
                n_rpt_xgb += int(scored['rpt_agrees_with_xgb'] or 0)
                xgb_pred = int(scored['xgb_is_anomaly'])
                rpt_pred = int(scored['rpt_is_anomaly'])
                if (
                    investigator is not None
                    and n_agent_reports < args.max_agent_reports
                    and should_run_agent(args.agent_trigger, xgb_pred, rpt_pred)
                ):
                    try:
                        report = investigator.investigate(build_agent_event(scored))
                        n_agent_reports += 1
                        record['agent_status'] = 'ok'
                        print(f'--- agent report {n_agent_reports}/{args.max_agent_reports} ---')
                        print(report)
                    except Exception as agent_exc:
                        record['agent_status'] = (
                            f'error: {type(agent_exc).__name__}: {agent_exc}'
                        )
                elif investigator is not None:
                    if n_agent_reports >= args.max_agent_reports:
                        record['agent_status'] = 'skipped_max'
                    else:
                        record['agent_status'] = 'skipped_trigger'
            writer.writerow(record)
            if scored['isolation_alert'] or (i % args.print_every == 0):
                print(json.dumps(record))
            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f'Output saved: {out}')
    print(f'Stage1 IF alerts={n_stage1_alerts} | Stage2 calls={n_stage2}')
    if n_stage2:
        print(
            f'Stage2 agreement | '
            f'XGB vs IF={n_xgb_if / n_stage2:.1%} | '
            f'RPT vs IF={n_rpt_if / n_stage2:.1%} | '
            f'RPT vs XGB={n_rpt_xgb / n_stage2:.1%}'
        )
    if args.agents:
        print(f'Agent reports={n_agent_reports} (max={args.max_agent_reports}) '
              f'-> {agent_reports_path}')
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument('--cooldown', type=int, default=30,
                        help='Minimum test rows between stage-2 (XGB+RPT) calls')
    parser.add_argument('--sleep', type=float, default=0)
    parser.add_argument('--print-every', type=int, default=100)
    parser.add_argument('--mock-rpt', action='store_true')
    parser.add_argument('--output', default='results/stream_predictions.csv')
    parser.add_argument('--agents', action='store_true')
    parser.add_argument('--agent-model', default='openai:gpt-4.1-mini')
    parser.add_argument('--max-agent-reports', type=int, default=3)
    parser.add_argument('--agent-trigger', choices=('confirmed', 'all'), default='confirmed')
    parser.add_argument('--agent-output', default='results/agent_reports.jsonl')
    args = parser.parse_args()
    if not (stage1_dir(args.artifacts) / 'meta.json').exists():
        parser.error('Stage1 artifacts missing; run stage1_fit_if.py first')
    if not (stage2_dir(args.artifacts) / 'meta.json').exists():
        parser.error('Stage2 artifacts missing; run stage2_fit_xgb_rpt_context.py first')
    if args.max_agent_reports < 1:
        parser.error('--max-agent-reports must be >= 1')
    run(args)
