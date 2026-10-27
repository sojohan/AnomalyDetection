"""Shared scoring session for stage3 CLI and the live score_server API."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline_common import (
    DEFAULT_ARTIFACTS,
    RPTPredictor,
    load_stage1_artifacts,
    load_stage2_artifacts,
    should_run_agent,
    stage1_dir,
    stage2_dir,
)

REPORT_SECTIONS = (
    'SUMMARY',
    'EVIDENCE',
    'POSSIBLE_CAUSES',
    'OPERATIONAL_IMPACT',
    'RECOMMENDED_CHECKS',
    'PRIORITY',
    'LIMITATIONS',
)


def parse_agent_report(report: str) -> dict[str, str]:
    """Split agent free-text into named sections for the UI."""
    text = report or ''
    sections = {name: '' for name in REPORT_SECTIONS}
    pattern = re.compile(
        r'^(SUMMARY|EVIDENCE|POSSIBLE_CAUSES|OPERATIONAL_IMPACT|'
        r'RECOMMENDED_CHECKS|PRIORITY|LIMITATIONS)\s*:\s*',
        re.MULTILINE | re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        sections['SUMMARY'] = text.strip()
        return sections
    for i, match in enumerate(matches):
        name = match.group(1).upper()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[name] = text[start:end].strip()
    return sections


@dataclass
class ScoringSession:
    """Loads stage1/stage2 artifacts and scores test rows with IF gate + XGB/RPT."""

    features: list[str]
    threshold: float
    test: pd.DataFrame
    imputer: Any
    if_model: Any
    xgb: Any
    rpt: RPTPredictor
    mock_rpt: bool
    cooldown: int
    meta: dict[str, Any]
    _last_call: int = field(default=-1, init=False)
    _row_index: int = field(default=0, init=False)

    @classmethod
    def load(
        cls,
        artifacts: Path | str = DEFAULT_ARTIFACTS,
        *,
        mock_rpt: bool = True,
        cooldown: int = 1,
    ) -> ScoringSession:
        artifacts = Path(artifacts)
        s1_path = stage1_dir(artifacts)
        s2_path = stage2_dir(artifacts)
        if not (s1_path / 'meta.json').exists():
            raise FileNotFoundError(f'Stage1 artifacts missing: {s1_path}')
        if not (s2_path / 'meta.json').exists():
            raise FileNotFoundError(f'Stage2 artifacts missing: {s2_path}')
        s1 = load_stage1_artifacts(s1_path)
        s2 = load_stage2_artifacts(s2_path, s1['features'])
        rpt_features = list(s2['meta'].get('rpt_features') or s1['features'])
        rpt = RPTPredictor(
            s2['context'], rpt_features, mock_rpt,
            s1['threshold'], s1['imputer'], s1['if_model'],
        )
        return cls(
            features=s1['features'],
            threshold=s1['threshold'],
            test=s1['test'],
            imputer=s1['imputer'],
            if_model=s1['if_model'],
            xgb=s2['xgb'],
            rpt=rpt,
            mock_rpt=mock_rpt,
            cooldown=cooldown,
            meta={
                'features': s1['features'],
                'rpt_features': rpt_features,
                'threshold': s1['threshold'],
                'n_test': len(s1['test']),
                'mock_rpt': mock_rpt,
                'cooldown': cooldown,
                'stage1': s1['meta'],
                'stage2': s2['meta'],
            },
        )

    def reset(self) -> None:
        self._last_call = -self.cooldown
        self._row_index = 0

    def score_row(self, row: pd.Series, index: int | None = None) -> dict[str, Any]:
        """Score one test row. Returns a JSON-serializable record."""
        i = self._row_index if index is None else index
        if index is None:
            self._row_index += 1

        vector = self.imputer.transform(
            pd.DataFrame([row[self.features]], columns=self.features)
        )
        score = float(self.if_model.score_samples(vector)[0])
        alert = score < self.threshold
        sensors = {
            col: (float(row[col]) if pd.notna(row[col]) else None)
            for col in self.features
            if col in row.index
        }
        actual_current = sensors.get('Motor_current')
        record: dict[str, Any] = {
            'index': i,
            'timestamp': str(row['timestamp']) if 'timestamp' in row.index else str(i),
            'sensors': sensors,
            'actual_current': actual_current,
            'isolation_score': round(score, 5),
            'isolation_alert': int(alert),
            'xgb_is_anomaly': None,
            'xgb_agrees_with_if': None,
            'rpt_is_anomaly': None,
            'rpt_agrees_with_if': None,
            'rpt_agrees_with_xgb': None,
            'rpt_status': 'not_called',
            'stage2_called': False,
            'agent_candidate': False,
        }
        if alert and i - self._last_call >= self.cooldown and row[self.features].notna().all():
            self._last_call = i
            xgb_pred = int(self.xgb.predict(vector)[0])
            record['xgb_is_anomaly'] = xgb_pred
            record['xgb_agrees_with_if'] = int(xgb_pred == int(alert))
            record['stage2_called'] = True
            try:
                feat = {col: float(row[col]) for col in self.rpt.features}
                rpt_pred = self.rpt.predict(feat)
                record['rpt_is_anomaly'] = rpt_pred
                record['rpt_agrees_with_if'] = int(rpt_pred == int(alert))
                record['rpt_agrees_with_xgb'] = int(rpt_pred == xgb_pred)
                record['rpt_status'] = 'mock' if self.mock_rpt else 'live'
                record['agent_candidate'] = True
            except Exception as exc:
                record['rpt_status'] = f'error: {type(exc).__name__}: {exc}'
        return record


def build_agent_event(record: dict[str, Any]) -> dict[str, Any]:
    return {
        'timestamp': record['timestamp'],
        'sensors': record.get('sensors') or {},
        'isolation_score': record['isolation_score'],
        'isolation_alert': record['isolation_alert'],
        'xgb_is_anomaly': record.get('xgb_is_anomaly'),
        'rpt_is_anomaly': record.get('rpt_is_anomaly'),
    }
