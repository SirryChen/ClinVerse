from __future__ import annotations

import json
from pathlib import Path


def _tool_trace_text(record: dict) -> str:
    lines: list[str] = []
    for turn in record.get("transcript", {}).get("turns", []):
        fn_name = turn.get("function", "")
        args = json.dumps(turn.get("args", {}), ensure_ascii=False)
        response = (turn.get("response") or "").strip()
        if len(response) > 400:
            response = response[:400] + " ..."
        lines.append(f"- tool={fn_name} args={args}")
        lines.append(f"  response={response}")
    return "\n".join(lines)


def _failure_items(record: dict) -> list[dict]:
    scores = record["scores"]
    items: list[dict] = []
    if scores.get("information_collection_score", scores.get("history_collection_score", 0)) < 0.25:
        items.append(
            {
                "title": "Insufficient baseline history collection",
                "description": "The doctor did not collect enough patient-known history before narrowing the diagnosis.",
                "content": "Ask explicitly about past medical history, medications, allergies, and social history before relying on tests.",
                "relation_to_skill": "Add a mandatory early-turn checklist for SAMPLE-style baseline history collection.",
            }
        )
    if scores.get("abnormal_test_coverage", 0) < 0.25:
        items.append(
            {
                "title": "Missed abnormal objective evidence",
                "description": "The doctor failed to surface key abnormal tests available in the case.",
                "content": "Order narrower but diagnosis-discriminative tests, then revisit abnormal findings with follow-up questions or confirmatory workup.",
                "relation_to_skill": "Strengthen the test-ordering heuristic so abnormal evidence is less likely to be skipped.",
            }
        )
    if scores.get("diagnosis_score", 0) <= 0:
        items.append(
            {
                "title": "Final diagnosis was outside the correct ICD block",
                "description": "The submitted diagnosis did not match the ground-truth disease family.",
                "content": "Keep the leading differential aligned with the strongest findings and convert the final hypothesis into the closest ICD-10 category before finalizing.",
                "relation_to_skill": "Add a pre-finalization verification step that checks whether history, exam, and tests all support the same disease family.",
            }
        )
    if "max_turns_reached" in record.get("tags", []):
        items.append(
            {
                "title": "Turn budget exhausted before convergence",
                "description": "The episode ended due to turn limit instead of a confident earlier convergence.",
                "content": "Avoid broad exploratory questioning once the top differential narrows; summarize evidence and commit sooner.",
                "relation_to_skill": "Add explicit turn-budget management guidance so the agent transitions from exploration to commitment earlier.",
            }
        )
    if not items:
        items.append(
            {
                "title": "General diagnostic inefficiency",
                "description": "The trajectory underperformed without a single dominant failure mode.",
                "content": "Use a tighter history -> exam -> targeted tests -> synthesis loop and remove redundant tool calls.",
                "relation_to_skill": "Refine the general workflow guidance for iterative but economical diagnosis.",
            }
        )
    return items


def _success_items(record: dict) -> list[dict]:
    tool_counts = record.get("tool_counts", {})
    items: list[dict] = []
    if tool_counts.get("ask_patient", 0) > 0:
        items.append(
            {
                "title": "History-first diagnostic framing",
                "description": "The doctor elicited patient-known context before committing to a workup.",
                "content": "Start with focused HPI and baseline history to shape the first differential and avoid low-yield testing.",
            }
        )
    if tool_counts.get("order_tests", 0) > 0:
        items.append(
            {
                "title": "Targeted test selection",
                "description": "The doctor used objective testing as a discriminator rather than ordering a vague blanket panel.",
                "content": "Order only the tests that can confirm or separate the current top hypotheses, then interpret them before ordering more.",
            }
        )
    if record["scores"].get("diagnosis_score", 0) >= 1.0:
        items.append(
            {
                "title": "Evidence-aligned finalization",
                "description": "The final diagnosis landed in the correct ICD block with evidence consistent with the transcript.",
                "content": "Before finalizing, reconcile the leading diagnosis with the highest-value findings and submit the closest ICD-10 code available.",
            }
        )
    if not items:
        items.append(
            {
                "title": "Reusable diagnosis loop",
                "description": "The trajectory shows a generally effective interactive diagnostic workflow.",
                "content": "Cycle through hypothesis formation, focused evidence gathering, and explicit synthesis until one diagnosis clearly dominates.",
            }
        )
    return items


def write_error_report(record: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Case {record['case_id']}",
        "",
        "## Metadata",
        f"- source: {record.get('source', 'unknown')}",
        f"- diagnosis_score: {record['scores'].get('diagnosis_score', 0)}",
        f"- information_collection_score: {record['scores'].get('information_collection_score', record['scores'].get('history_collection_score', 0))}",
        f"- abnormal_test_coverage: {record['scores'].get('abnormal_test_coverage', 0)}",
        "",
        "## Tool Trace",
        _tool_trace_text(record),
        "",
    ]
    for idx, item in enumerate(_failure_items(record), start=1):
        lines.extend(
            [
                f"# Failure Cause Item {idx}",
                "",
                "## Title",
                item["title"],
                "",
                "## Description",
                item["description"],
                "",
                "## Content",
                item["content"],
                "",
                "## Relation to Skill",
                item["relation_to_skill"],
                "",
            ]
        )
    report_path = out_dir / "analysis_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_success_report(record: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Case {record['case_id']}",
        "",
        "## Metadata",
        f"- source: {record.get('source', 'unknown')}",
        f"- diagnosis_score: {record['scores'].get('diagnosis_score', 0)}",
        "",
        "## Tool Trace",
        _tool_trace_text(record),
        "",
    ]
    for idx, item in enumerate(_success_items(record), start=1):
        lines.extend(
            [
                f"# Success Memory Item {idx}",
                "",
                "## Title",
                item["title"],
                "",
                "## Description",
                item["description"],
                "",
                "## Content",
                item["content"],
                "",
            ]
        )
    report_path = out_dir / "success_analysis.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
