#!/usr/bin/env python3
"""Probe a doctor model's diagnosis at regular prefixes of eval trajectories.

The script reads only ``transcript.initial_visible_info`` and turns up to each
sampled checkpoint. With ``--step 2``, checkpoints are 0, 1, 3, 5, ... . It
never puts the trace's final answer, scores, or ground truth in the model
prompt. Those fields may be copied to the result record for later analysis,
but are kept separate from ``messages``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Iterable
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
CLINVERSE_ROOT = REPO_ROOT
sys.path.insert(0, str(CLINVERSE_ROOT))

from core.config import load_yaml_config  # noqa: E402

DEFAULT_RESULTS_ROOT = CLINVERSE_ROOT / "data" / "eval_results"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results"

SYSTEM_PROMPT = """你是一名临床诊断医生。请只根据提供的初始病情和既往医患交互，判断当前最可能的主要疾病。
不要声称看到了未提供的检查、后续对话、最终诊断或出院信息。请简要说明诊断及依据。

返回 JSON 对象，且只包含以下字段：
{"primary_diagnosis": "<最可能的主要疾病>", "icd10_code": "<若能判断则填写 ICD-10-CM，否则为空字符串>", "rationale": "<简短依据>"}"""


def load_config(path: Path) -> dict:
    return load_yaml_config(path)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def candidate_run_dirs(results_root: Path) -> Iterable[Path]:
    """Yield only directories that look like a batch-evaluation run."""
    if not results_root.is_dir():
        return []
    return sorted(
        path for path in results_root.iterdir()
        if path.is_dir() and any(path.glob("repeat_*/metadata.json"))
    )


def resolve_run_dir(results_root: Path, provider: str, run_id: str | None) -> Path:
    """Resolve an explicit run, or find the provider's most complete run.

    A run is associated with a doctor provider through its per-repeat metadata,
    which avoids depending on inconsistent historical run-id naming schemes.
    """
    if run_id:
        path = Path(run_id)
        resolved = path if path.is_absolute() else results_root / path
        if not resolved.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {resolved}")
        return resolved

    # Conventional baseline run IDs use either hyphens or underscores.  Prefer
    # this direct model-to-folder mapping over a larger held-out/ablation run.
    provider_stem = provider.removesuffix("-vllm")
    conventional_names = (
        f"mimic100_{provider_stem}",
        f"mimic100_{provider_stem.replace('-', '_')}",
    )
    for name in conventional_names:
        path = results_root / name
        metadata_paths = sorted(path.glob("repeat_*/metadata.json")) if path.is_dir() else []
        if any(load_json(item).get("doctor_provider") == provider for item in metadata_paths):
            return path

    matches: list[tuple[int, str, Path]] = []
    for run_dir in candidate_run_dirs(results_root):
        metadata_files = sorted(run_dir.glob("repeat_*/metadata.json"))
        metadata = []
        for path in metadata_files:
            try:
                metadata.append(load_json(path))
            except (OSError, json.JSONDecodeError):
                continue
        if not any(row.get("doctor_provider") == provider for row in metadata):
            continue
        trace_count = sum(1 for _ in run_dir.glob("repeat_*/traces/*.json"))
        matches.append((trace_count, run_dir.name, run_dir))
    if not matches:
        raise FileNotFoundError(
            f"No eval run with doctor_provider={provider!r} under {results_root}. "
            "Pass --run-id to select a trajectory folder manually."
        )
    # Prefer the largest usable trajectory set; use the name for deterministic ties.
    return max(matches, key=lambda item: (item[0], item[1]))[2]


def iter_trace_paths(run_dir: Path, repeat: str | None) -> list[Path]:
    if repeat:
        repeat_dir = run_dir / repeat
        if not repeat_dir.is_dir():
            raise FileNotFoundError(f"Repeat directory does not exist: {repeat_dir}")
        paths = sorted((repeat_dir / "traces").glob("*.json"))
    else:
        paths = sorted(run_dir.glob("repeat_*/traces/*.json"))
        paths.extend(sorted((run_dir / "traces").glob("*.json")))
    if not paths:
        raise FileNotFoundError(f"No trajectory JSON files found under {run_dir}")
    return paths


def concise_action(turn: dict) -> str:
    """Render the doctor action without hidden metadata or tool internals."""
    function = turn.get("function", "")
    args = turn.get("args") or {}
    if function == "ask_patient":
        return str(args.get("question", "医生向患者提问。"))
    if function == "request_physical_exam":
        components = "、".join(map(str, args.get("components") or []))
        return f"医生请求体格检查：{components or '未说明项目'}。"
    if function == "order_tests":
        tests = "、".join(map(str, args.get("tests") or []))
        return f"医生开具检查：{tests or '未说明项目'}。"
    if function == "order_imaging":
        studies = "、".join(map(str, args.get("studies") or args.get("imaging") or []))
        return f"医生申请影像检查：{studies or '未说明项目'}。"
    return f"医生执行操作：{function or '未说明'}。"


def build_history(transcript: dict, checkpoint: int) -> tuple[str, int]:
    """Return visible dialogue through ``checkpoint`` and its included turn count."""
    visible = transcript.get("initial_visible_info") or {}
    initial = "\n".join([
        "## 初始可见信息",
        f"- 年龄：{visible.get('age', '未记录')}",
        f"- 性别：{visible.get('gender', '未记录')}",
        f"- 入院类型：{visible.get('admission_type', '未记录')}",
        f"- 生命体征/初始查体：{visible.get('vital_signs', '未记录')}",
        f"- 主诉：{visible.get('chief_complaint', '未记录')}",
    ])
    rendered = [initial, "## 历史医患交互"]
    included = 0
    for turn in transcript.get("turns") or []:
        turn_number = int(turn.get("turn", 0))
        if turn_number > checkpoint:
            continue
        rendered.append(f"\n### 第 {turn_number} 轮\n医生：{concise_action(turn)}\n患者/系统：{turn.get('response', '')}")
        included += 1
    if not included:
        rendered.append("（尚无历史交互。）")
    return "\n".join(rendered), included


def probe_checkpoints(interval: int, max_turns: int = 20) -> list[int]:
    """Return 0, 1, 1+interval, ... through the evaluation turn cap."""
    return [0, *range(1, max_turns + 1, interval)]


def checkpoints_for_trace(transcript: dict, interval: int) -> tuple[list[int], int]:
    """Return probes actually needed for one trace and its terminal turn.

    Checkpoints 0 and 1 are always retained.  Later checkpoints after an
    episode has terminated are deliberately omitted: the plotting stage
    carries the last available result forward instead of re-querying an
    unchanged history.
    """
    turns = transcript.get("turns") or []
    observed_last_turn = max((int(turn.get("turn", 0)) for turn in turns), default=0)
    terminal_turn = int(transcript.get("total_turns") or observed_last_turn)
    checkpoints = [0, 1]
    checkpoints.extend(
        checkpoint for checkpoint in probe_checkpoints(interval)
        if checkpoint > 1 and checkpoint <= terminal_turn
    )
    return sorted(set(checkpoints)), terminal_turn


def select_trajectories(
    trace_paths: list[Path], limit: int | None, sample_seed: int,
) -> tuple[list[Path], dict]:
    """Randomly sample from every candidate trajectory."""
    shuffled = list(trace_paths)
    random.Random(sample_seed).shuffle(shuffled)
    selected = shuffled[:limit] if limit is not None else shuffled
    return selected, {
        "selection_policy": "random_all_trajectories",
        "n_candidates": len(shuffled),
        "n_requested": limit if limit is not None else len(shuffled),
        "n_selected": len(selected),
    }


def parse_response(content: str) -> dict:
    """Preserve raw output and opportunistically unpack the requested JSON."""
    raw = (content or "").strip()
    cleaned = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        data = {}
    return {
        "raw_response": raw,
        "primary_diagnosis": str(data.get("primary_diagnosis", "")).strip(),
        "predicted_icd10": str(data.get("icd10_code", "")).strip().upper(),
        "rationale": str(data.get("rationale", "")).strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, help="ClinVerse config llm_providers key for the doctor model.")
    parser.add_argument("--step", type=int, required=True, help="Sampling interval: step=2 probes 0, 1, 3, 5, ... (valid range: 1–20).")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of trajectories to probe (default: all).")
    parser.add_argument(
        "--sample-seed", type=int, default=20260722,
        help="Random trajectory sampling seed (default: 20260722).",
    )
    parser.add_argument("--run-id", help="Optional eval run-id or absolute trajectory folder; overrides automatic provider lookup.")
    parser.add_argument("--repeat", help="Optional repeat directory, e.g. repeat_01 (default: all repeats).")
    parser.add_argument("--config", type=Path, default=CLINVERSE_ROOT / "config" / "default.yaml")
    parser.add_argument("--eval-results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--out", type=Path, help="Output JSONL path (default: experiment-local results directory).")
    parser.add_argument("--resume", action="store_true", help="Skip trajectory records already present in --out.")
    parser.add_argument("--retry-failed", action="store_true", help="With --resume, retry records that previously failed.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print selected trajectories without calling the model.")
    args = parser.parse_args()

    if not 1 <= args.step <= 20:
        parser.error("--step must be an interval between 1 and 20.")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")

    config = load_config(args.config)
    provider = config.get("llm_providers", {}).get(args.provider)
    if not provider:
        parser.error(f"Provider not found in config: {args.provider}")
    if not provider.get("doctor_model") or not provider.get("base_url"):
        parser.error(f"Provider {args.provider!r} needs doctor_model and base_url.")

    run_dir = resolve_run_dir(args.eval_results_root, args.provider, args.run_id)
    trace_paths = iter_trace_paths(run_dir, args.repeat)
    trace_paths, selection = select_trajectories(trace_paths, args.limit, args.sample_seed)
    checkpoints = probe_checkpoints(args.step)
    out = args.out or DEFAULT_OUTPUT_ROOT / f"{run_dir.name}_{args.repeat or 'all'}_every{args.step}_random_carryforward.jsonl"

    if args.dry_run:
        print(json.dumps({"run_dir": str(run_dir), **selection, "sampling_interval": args.step, "checkpoints": checkpoints, "sample_seed": args.sample_seed, "out": str(out)}, ensure_ascii=False, indent=2))
        return

    existing = read_jsonl(out) if args.resume else []
    completed = {
        (row.get("trajectory_id"), row.get("step"))
        for row in existing
        if row.get("probe_mode") == "interval_carry_forward" and not row.get("error")
    }
    failed = {
        (row.get("trajectory_id"), row.get("step"))
        for row in existing
        if row.get("probe_mode") == "interval_carry_forward" and row.get("error")
    }
    client = OpenAI(api_key=provider["api_key"], base_url=provider["base_url"], timeout=provider.get("timeout", 180))
    out.parent.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[Path, int, int]] = []
    for path in trace_paths:
        trajectory_id = str(path.relative_to(run_dir))
        try:
            transcript = (load_json(path).get("transcript") or {})
            trace_checkpoints, terminal_turn = checkpoints_for_trace(transcript, args.step)
        except Exception:
            # The call loop will emit a durable error record for this path.
            trace_checkpoints, terminal_turn = [0], 0
        for checkpoint in trace_checkpoints:
            key = (trajectory_id, checkpoint)
            if args.resume and (key in completed or (key in failed and not args.retry_failed)):
                continue
            pending.append((path, checkpoint, terminal_turn))

    with out.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for path, checkpoint, terminal_turn in tqdm(pending, desc=f"diagnosis probe every {args.step} turns", unit="probe"):
            trajectory_id = str(path.relative_to(run_dir))
            try:
                trace = load_json(path)
                transcript = trace.get("transcript") or {}
                history, included_turns = build_history(transcript, checkpoint)
                # Scores stay outside messages. They are persisted solely for offline analysis.
                scores = trace.get("scores") or {}
                response = client.chat.completions.create(
                    model=provider["doctor_model"],
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": history}],
                    temperature=0,
                    max_tokens=256,
                    extra_body=provider.get("extra_body"),
                )
                record = {
                    "trajectory_id": trajectory_id,
                    "source_run_id": run_dir.name,
                    "source_trace": str(path),
                    "case_id": trace.get("case_id") or transcript.get("case_id"),
                    "provider": args.provider,
                    "selection_policy": selection["selection_policy"],
                    "model": provider["doctor_model"],
                    "probe_mode": "interval_carry_forward",
                    "sampling_interval": args.step,
                    "step": checkpoint,
                    "trajectory_terminal_turn": terminal_turn,
                    "included_turns": included_turns,
                    "history": history,
                    **parse_response(response.choices[0].message.content),
                    "ground_truth_icd10_for_analysis_only": scores.get("ground_truth_icd10", ""),
                    "ground_truth_for_analysis_only": scores.get("ground_truth", ""),
                }
            except Exception as exc:  # Keep the batch resumable when one call fails.
                record = {"trajectory_id": trajectory_id, "source_run_id": run_dir.name, "provider": args.provider, "selection_policy": selection["selection_policy"], "probe_mode": "interval_carry_forward", "sampling_interval": args.step, "step": checkpoint, "trajectory_terminal_turn": terminal_turn, "error": str(exc)}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    print(json.dumps({"run_dir": str(run_dir), **selection, "sampling_interval": args.step, "checkpoints": checkpoints, "sample_seed": args.sample_seed, "n_attempted_probes": len(pending), "out": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
