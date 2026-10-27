"""Event-driven Deep Agents investigation of MetroPT-3 anomalies.

LLM outputs are recommendations, not validated diagnoses. No external actions.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from failure_knowledge import reference_for_event


SYSTEM = """You are the supervisor of an industrial compressor anomaly investigation.
The input is a single JSON event with sensor readings, an Isolation Forest alert,
and two classifiers (XGBoost and SAP RPT) that were trained on Isolation Forest
pseudo-labels, NOT on verified failures. Their agreement is not proof of failure.
Delegate to diagnosis, operational impact, and maintenance subagents, then
produce a SHORT scannable report for an operator dashboard. Do not invent
thresholds, manuals, maintenance history, failure probabilities, costs, or
causal certainty. Historical failure intervals from the supplied PDF are only
available when already completed as of the event timestamp. Never claim that a
prior air leak establishes the cause of a new alert. No tool can actuate
equipment. Never recommend bypassing safety interlocks. For potentially
hazardous situations, defer to plant safety procedures and qualified personnel.
If model predictions disagree, clearly state uncertainty.

Hard length rules — operators skim this:
- SUMMARY: 1-2 sentences only (what happened + confidence).
- EVIDENCE: 3-5 bullets; cite concrete sensor values and Isolation Forest score
  only. Do NOT mention that XGBoost and/or SAP RPT flagged the event (that is
  already known — agents only run on confirmed XGB+RPT agreement).
- POSSIBLE_CAUSES: 2-4 bullets; label each as hypothesis, not fact.
- OPERATIONAL_IMPACT: 1-3 bullets; qualitative only.
- RECOMMENDED_CHECKS: 3-5 short numbered actions (non-invasive).
- PRIORITY: exactly one of review | urgent_review | monitor
- LIMITATIONS: 2-3 short bullets.
Prefer "- " bullets. No long paragraphs. No repeated caveats across sections.

PRIORITY calibration (decision support only — never a shutdown order):
- monitor: use ONLY when models disagree, evidence is weak/ambiguous, or the
  pattern looks like a known benign / unloaded operating state.
- review: default when Isolation Forest alerts AND XGBoost and RPT both flag
  anomaly (confirmed). Do NOT choose monitor in that case — even if you note
  uncertainty from pseudo-labels, confirmed agreement warrants human review.
- urgent_review: only if sensors suggest rapidly worsening or safety-relevant
  risk beyond a routine confirmed anomaly (still defer actuation to plant SOPs).
Never write "high confidence" together with PRIORITY: monitor.

Use this exact structure in the final response:
SUMMARY: ...
EVIDENCE:
- ...
POSSIBLE_CAUSES:
- ...
OPERATIONAL_IMPACT:
- ...
RECOMMENDED_CHECKS:
1. ...
PRIORITY: review | urgent_review | monitor
LIMITATIONS:
- ...
"""

DIAGNOSIS = """Investigate the anomaly using ONLY supplied sensor readings and
historical comparisons. Consult the provided sensor definitions and prior
completed failure reports. Identify possible (not confirmed) causes, and which
additional evidence would distinguish them. In EVIDENCE, focus on sensors and
the Isolation Forest alert — do not restate XGBoost/RPT agreement. Do not
assert that the compressor has failed or that a specific reading exceeds a
manufacturer limit.
"""

IMPACT = """Assess potential operational implications qualitatively.
You have no production schedules, redundancy information, failure probabilities,
maintenance costs, or equipment criticality. Say what cannot be quantified.
Identify which plant facts are needed to prioritize intervention.
"""

MAINTENANCE = """Propose a short prioritized checklist of non-invasive checks
and operator review actions. Do not give hazardous repair instructions or
recommend shutting down a machine based only on an ML alert. Escalation should
follow local SOPs and qualified maintenance personnel. Distinguish inspection
from confirmed repair. Avoid inventing OEM specifications.
"""


def build_agent(model: str):
    """Build a supervisor with three specialist Deep Agents subagents."""
    try:
        from deepagents import create_deep_agent
    except ImportError as exc:
        raise RuntimeError('Install deepagents and langchain-core') from exc

    @tool
    def explain_detector_semantics() -> str:
        """Return reliable interpretation of the ML signals used in this demo."""
        return (
            'Isolation Forest: lower score_samples means more anomalous; '
            'the alert threshold is a training-set quantile. XGBoost and '
            'SAP RPT are trained on IF-generated pseudo-labels. Agreement '
            'measures imitation/consistency, not fault detection accuracy. '
            'No official failure labels are used in this workflow.'
        )

    subagents = [
        {'name': 'diagnosis', 'description': 'Investigate sensor anomalies and plausible causes',
         'system_prompt': DIAGNOSIS, 'tools': [explain_detector_semantics]},
        {'name': 'impact', 'description': 'Assess possible operational impact and missing business context',
         'system_prompt': IMPACT, 'tools': []},
        {'name': 'maintenance', 'description': 'Recommend safe inspections and maintenance follow-up',
         'system_prompt': MAINTENANCE, 'tools': []},
    ]
    # Passing a model string lets LangChain initialize a supported chat provider.
    return create_deep_agent(model=model, tools=[explain_detector_semantics],
                             subagents=subagents, system_prompt=SYSTEM)


def _content_text(message: Any) -> str:
    content = getattr(message, 'content', '')
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return '\n'.join(str(part.get('text', '')) if isinstance(part, dict) else str(part)
                         for part in content)
    return str(content)


class AnomalyInvestigator:
    def __init__(self, model: str, output: str):
        self.agent = build_agent(model)
        self.path = Path(output)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def investigate(self, event: dict[str, Any]) -> str:
        prompt = (
            'Investigate this anomaly event. Delegate to all three specialists '
            'and distinguish observed facts from hypotheses.\n'
            + json.dumps({'event':event, 'historical_reference':reference_for_event(event.get('timestamp'))}, ensure_ascii=False, default=str)
        )
        result = self.agent.invoke({'messages': [{'role': 'user', 'content': prompt}]})
        messages = result.get('messages', [])
        report = _content_text(messages[-1]) if messages else 'No agent output'
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'event': event, 'report': report},
                               ensure_ascii=False, default=str) + '\n')
        return report
