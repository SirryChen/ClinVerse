#!/usr/bin/env python3
"""Batch evaluation: run simulations over many cases and aggregate scores."""

import argparse
import json
import random
import statistics
import subprocess
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from core.config import load_yaml_config
from env.task_stream import TaskStream
from simulation.session import SimulationSession
from evaluation.reporter import generate_report
from doctor_train.memrl import (
    add_or_replace_memory,
    format_memory_context,
    intent_from_case,
    load_memory_bank,
    reward_from_scores,
    retrieve_memories,
    save_memory_bank,
    update_memory_values,
)
from doctor_train.medagent_zero import (
    add_or_replace_record as medagent_zero_add_or_replace_record,
    format_context as medagent_zero_format_context,
    load_bank as medagent_zero_load_bank,
    query_from_case as medagent_zero_query_from_case,
    retrieve as medagent_zero_retrieve,
    save_bank as medagent_zero_save_bank,
)
from doctor_train.rollouts import build_rollout_record
from doctor_train.skillrl import SkillRLTemplateMemory


def load_config(path: str) -> dict:
    return load_yaml_config(path)


def resolve_config_path(config_path: str | Path, value: str) -> Path:
    base_dir = Path(config_path).resolve().parent.parent
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path)


def resolve_memrl_input_path(args) -> str | None:
    if args.memrl_memory:
        return args.memrl_memory
    if args.memrl_update_output and Path(args.memrl_update_output).exists():
        return args.memrl_update_output
    return None


def resolve_medagent_zero_input_path(args) -> str | None:
    if args.medagent_zero_bank:
        return args.medagent_zero_bank
    if args.medagent_zero_update_output and Path(args.medagent_zero_update_output).exists():
        return args.medagent_zero_update_output
    return None


def load_skillrl_memory(args):
    if not args.skillrl_bank:
        return None
    if args.skillrl_retrieval_mode != "template":
        raise ValueError("ClinVerse currently supports SkillRL template retrieval only.")
    return SkillRLTemplateMemory(
        skills_json_path=args.skillrl_bank,
        task_specific_top_k=args.skillrl_task_specific_top_k,
    )


def load_trace2skill_context(skill_path: str | None) -> str | None:
    """Load a static Trace2Skill clinical skill for doctor-prompt injection."""
    if not skill_path:
        return None
    path = Path(skill_path)
    if path.is_dir():
        path = path / "SKILL.md"
    if not path.is_file():
        raise FileNotFoundError(
            "Trace2Skill skill must be a SKILL.md file or a directory containing one: "
            f"{skill_path}"
        )
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Trace2Skill skill is empty: {path}")
    return "## Trace2Skill clinical guidance\n\n" + content


def skillrl_query_from_case(case: dict) -> str:
    return (
        "Clinical diagnosis task: collect a focused history, order targeted examinations "
        "and tests, then synthesize a diagnosis. "
        + intent_from_case(case)
    )


def load_jsonl_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_existing_errors(results_dir: Path) -> list[dict]:
    errors_jsonl = results_dir / "errors.jsonl"
    if errors_jsonl.exists():
        return load_jsonl_records(errors_jsonl)
    errors_json = results_dir / "errors.json"
    if errors_json.exists():
        with errors_json.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    return []


def seed_errors_jsonl(errors_jsonl_path: Path, errors: list[dict]) -> None:
    if errors_jsonl_path.exists() or not errors:
        return
    with errors_jsonl_path.open("w", encoding="utf-8") as f:
        for error in errors:
            f.write(json.dumps(error, ensure_ascii=False) + "\n")


def load_excluded_case_ids(sources: list[str]) -> tuple[set[str], list[str]]:
    """Load case IDs from rollout JSONL files or directories containing them."""
    excluded: set[str] = set()
    resolved_paths: list[str] = []
    for source in sources:
        path = Path(source)
        path = path / "rollouts.jsonl" if path.is_dir() else path
        if not path.is_file():
            raise FileNotFoundError(
                f"Exclusion source must be a rollout JSONL file or directory containing "
                f"rollouts.jsonl: {source}"
            )
        records = load_jsonl_records(path)
        excluded.update(str(record["case_id"]) for record in records if record.get("case_id"))
        resolved_paths.append(str(path.resolve()))
    return excluded, resolved_paths


SUMMARY_METRICS = [
    ("Info Collection", "information_collection_score"),
    ("Test Order Coverage", "test_order_coverage"),
    ("Abnormal Test Coverage", "abnormal_test_coverage"),
    ("Diagnosis Category Accuracy", "diagnosis_category_accuracy"),
    ("Diagnosis Subtype Accuracy", "diagnosis_subtype_accuracy"),
]


def round_scores(scores: dict) -> dict:
    """Return a copy of evaluation scores rounded to four decimal places."""
    return {
        key: round(value, 4) if isinstance(value, float) else value
        for key, value in scores.items()
    }


def remove_cli_options(argv: list[str], options: set[str]) -> list[str]:
    """Remove value-taking options from an argv list, preserving all others."""
    filtered = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        option = arg.split("=", 1)[0]
        if option in options:
            i += 1 if "=" in arg else 2
            continue
        filtered.append(arg)
        i += 1
    return filtered


def load_completed_repeat_summary(repeat_dir: Path, requested_n: int | None) -> dict | None:
    """Return a completed repeat's summary, or None when it must be resumed."""
    summary_path = repeat_dir / "summary.json"
    if not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not summary.get("metrics"):
        return None

    metadata_path = repeat_dir / "metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("target_met") is True:
                return summary
            if metadata.get("target_met") is False:
                return None
        except (OSError, json.JSONDecodeError):
            pass

    # Older result directories may not have target_met in metadata. A complete
    # summary with at least the requested number of cases is still reusable.
    if requested_n is None or int(summary.get("n", 0)) >= requested_n:
        return summary
    return None


def repeat_summary_record(repeat_number: int, repeat_id: str, repeat_dir: Path, summary: dict) -> dict:
    return {
        "repeat": repeat_number,
        "run_id": repeat_id,
        "output_dir": str(repeat_dir.resolve()),
        "n": summary.get("n", 0),
        "metrics": summary.get("metrics", {}),
    }


def run_repeated_evaluations(args) -> None:
    """Run independent evaluations and save an aggregate of their summaries."""
    cfg = load_config(args.config)
    group_run_id = args.run_id or uuid.uuid4().hex[:8]
    results_dir = (
        Path(args.output_root)
        if args.output_root
        else resolve_config_path(args.config, cfg["paths"]["eval_results"]) / group_run_id
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    child_argv = remove_cli_options(
        sys.argv[1:],
        {"--repeats", "--output-root", "--run-id", "--sample-seed"},
    )
    repeat_summaries = []
    for repeat_index in range(args.repeats):
        repeat_number = repeat_index + 1
        repeat_id = f"{group_run_id}_repeat_{repeat_number:02d}"
        repeat_dir = results_dir / f"repeat_{repeat_number:02d}"
        completed_summary = load_completed_repeat_summary(repeat_dir, args.n)
        if completed_summary is not None:
            print(f"\n=== REPEAT {repeat_number}/{args.repeats}: already complete; reusing ===", flush=True)
            repeat_summaries.append(
                repeat_summary_record(repeat_number, repeat_id, repeat_dir, completed_summary)
            )
            continue

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *child_argv,
            "--repeats", "1",
            "--run-id", repeat_id,
            "--output-root", str(repeat_dir),
        ]
        # With a supplied seed, each repeat is reproducible but samples a
        # different shuffle order. Without one, the normal fresh shuffle logic
        # remains unchanged.
        if args.sample_seed is not None:
            command.extend(["--sample-seed", str(args.sample_seed + repeat_index)])
        # When a repeat directory exists but is incomplete, resume its JSONL
        # records rather than starting that repeat from scratch.
        if (repeat_dir / "rollouts.jsonl").exists() and "--resume" not in child_argv:
            command.append("--resume")
        print(f"\n=== REPEAT {repeat_number}/{args.repeats} ===", flush=True)
        completed = subprocess.run(command)
        if completed.returncode:
            raise RuntimeError(f"Repeat {repeat_number} failed with exit code {completed.returncode}.")

        summary_path = repeat_dir / "summary.json"
        with summary_path.open(encoding="utf-8") as f:
            summary = json.load(f)
        repeat_summaries.append(
            repeat_summary_record(repeat_number, repeat_id, repeat_dir, summary)
        )

    metric_keys = sorted({
        key for summary in repeat_summaries for key in summary["metrics"]
    })
    metrics = {}
    for key in metric_keys:
        repeat_means = [
            summary["metrics"][key]["mean"]
            for summary in repeat_summaries
            if key in summary["metrics"]
        ]
        metrics[key] = {
            "mean": round(statistics.mean(repeat_means), 4),
            "std": round(statistics.stdev(repeat_means), 4) if len(repeat_means) > 1 else 0.0,
            "repeat_means": [round(value, 4) for value in repeat_means],
        }

    aggregate = {
        "repeats": args.repeats,
        "n_requested_per_repeat": args.n,
        "n": sum(summary["n"] for summary in repeat_summaries),
        "metrics": metrics,
        "repeat_summaries": repeat_summaries,
    }
    (results_dir / "summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n=== REPEATED RUN SUMMARY (repeats={args.repeats}) ===")
    for label, metric_key in SUMMARY_METRICS:
        stats = metrics.get(metric_key)
        if stats:
            print(f"  {label}: mean={stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"\nResults saved to {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation across multiple cases.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--cases-dir", default=None, help="Override cases dir")
    parser.add_argument("--n", type=int, default=None, help="Target number of successful cases (default); see --allow-fewer-successes.")
    parser.add_argument("--shuffle", action="store_true", help="Sample cases randomly instead of using index order.")
    parser.add_argument("--sample-seed", type=int, default=None, help="Random seed for --shuffle. Omit for a fresh random sample each run.")
    parser.add_argument(
        "--exclude-case-ids-from",
        action="append",
        default=[],
        metavar="ROLLOUTS_OR_DIR",
        help=(
            "Exclude case IDs found in a rollouts.jsonl file, or in rollouts.jsonl under "
            "the given directory. Repeat to exclude multiple training pools."
        ),
    )
    parser.add_argument("--run-id", default=None, help="Run identifier (default: random)")
    parser.add_argument("--output-root", default=None, help="Override output directory. Defaults to eval_results/run_id.")
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Run the full evaluation this many times and average per-run metrics (default: 1).",
    )
    parser.add_argument("--resume", action="store_true", help="Append to an existing output root and skip completed/failed cases.")
    parser.add_argument("--retry-failed", action="store_true", help="With --resume, retry cases already present in errors.json/errors.jsonl.")
    parser.add_argument(
        "--target-completed",
        action="store_true",
        default=True,
        help="Treat --n as the target number of successful evaluations (default).",
    )
    parser.add_argument(
        "--allow-fewer-successes",
        action="store_false",
        dest="target_completed",
        help="Treat --n as a fixed number of attempts instead of replacing failed cases.",
    )
    parser.add_argument("--only-usable", action="store_true", default=True,
                        help="Skip cases flagged as not usable")
    parser.add_argument("--source", default=None,
                        help="Only run cases from this dataset adapter (e.g. mimic-iv-ext, sicdb-ext). "
                             "Default: all sources mixed in --cases-dir.")
    parser.add_argument("--env-provider", default=None,
                        help="Deprecated shortcut: set both patient and lab providers at once. "
                             "Default: config's default_llm_provider.")
    parser.add_argument("--patient-provider", default=None,
                        help="llm_providers key driving the patient simulator. "
                             "Default: --env-provider or config's default_llm_provider.")
    parser.add_argument("--lab-provider", default=None,
                        help="llm_providers key driving missing-lab fallback prediction. "
                             "Default: --patient-provider.")
    parser.add_argument("--doctor-provider", default=None,
                        help="llm_providers key for the doctor agent under test. "
                             "Default: same as --patient-provider.")
    parser.add_argument("--max-turns", type=int, default=None,
                        help="Override simulation.max_turns for quick smoke tests.")
    parser.add_argument("--memrl-memory", default=None,
                        help="Existing Hospital-MemRL memory bank JSON.")
    parser.add_argument("--memrl-top-k", type=int, default=0,
                        help="Number of memories to inject into each doctor prompt.")
    parser.add_argument("--memrl-min-similarity", type=float, default=0.0)
    parser.add_argument("--memrl-alpha", type=float, default=0.1)
    parser.add_argument("--memrl-weight-similarity", type=float, default=0.5)
    parser.add_argument("--memrl-weight-q", type=float, default=0.5)
    parser.add_argument("--memrl-q-min-threshold", type=float, default=None)
    parser.add_argument("--memrl-update-output", default=None,
                        help="Write the updated memory bank after the run.")
    parser.add_argument("--memrl-add-new", action="store_true",
                        help="Add completed rollouts to the memory bank.")
    parser.add_argument("--skillrl-bank", default=None, help="LLM-distilled SkillRL skill bank JSON.")
    parser.add_argument("--skillrl-top-k", type=int, default=6, help="Number of general SkillRL skills to inject.")
    parser.add_argument("--skillrl-task-specific-top-k", type=int, default=3,
                        help="Number of task-specific SkillRL skills to inject.")
    parser.add_argument("--skillrl-retrieval-mode", choices=["template", "embedding"], default="template")
    parser.add_argument("--skillrl-embedding-model", default=None,
                        help="Embedding model path/ID; used only with --skillrl-retrieval-mode embedding.")
    parser.add_argument(
        "--trace2skill-skill", default=None,
        help="Trace2Skill SKILL.md file or directory containing SKILL.md to inject into the doctor prompt.",
    )
    parser.add_argument("--medagent-zero-bank", default=None,
                        help="MedAgent-Zero dual-library JSON to retrieve from.")
    parser.add_argument("--medagent-zero-stage", default="diagnosis",
                        choices=["examination", "diagnosis", "treatment"])
    parser.add_argument("--medagent-zero-top-k-records", type=int, default=0)
    parser.add_argument("--medagent-zero-top-k-experiences", type=int, default=0)
    parser.add_argument("--medagent-zero-min-similarity", type=float, default=0.0)
    parser.add_argument("--medagent-zero-update-output", default=None,
                        help="Write the updated MedAgent-Zero bank after the run.")
    parser.add_argument("--medagent-zero-add-new", action="store_true",
                        help="Add completed rollouts to the MedAgent-Zero dual library.")
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.repeats > 1:
        run_repeated_evaluations(args)
        return

    cfg = load_config(args.config)
    repo_root = Path(__file__).parent.parent

    cases_dir    = resolve_config_path(args.config, args.cases_dir or cfg["paths"]["cases"])
    run_id       = args.run_id or uuid.uuid4().hex[:8]
    results_dir  = (
        Path(args.output_root)
        if args.output_root
        else resolve_config_path(args.config, cfg["paths"]["eval_results"]) / run_id
    )
    traces_dir   = results_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    scores_path = results_dir / "scores.jsonl"
    rollouts_path = results_dir / "rollouts.jsonl"
    errors_jsonl_path = results_dir / "errors.jsonl"

    existing_records = load_jsonl_records(rollouts_path) if args.resume else []
    existing_errors = load_existing_errors(results_dir) if args.resume else []
    if args.resume:
        seed_errors_jsonl(errors_jsonl_path, existing_errors)
    completed_case_ids = {str(record.get("case_id")) for record in existing_records if record.get("case_id")}
    failed_case_ids = {str(error.get("case_id")) for error in existing_errors if error.get("case_id")}

    # Per-dataset adapter construction kwargs (data_dir override, etc.) — each
    # case picks its own dataset via core_admission.source, so this is no
    # longer a single global lab_catalog/radiology_catalog pair (see
    # simulation.session.SimulationSession.from_case_file).
    datasets_config = {
        name: {"data_dir": str((repo_root / cfg_entry["data_dir"]).resolve())}
        for name, cfg_entry in cfg.get("datasets", {}).items()
        if cfg_entry.get("data_dir")
    }

    excluded_case_ids, exclusion_sources = load_excluded_case_ids(args.exclude_case_ids_from)
    case_refs = list(TaskStream(cases_dir, only_usable=args.only_usable, source=args.source))
    excluded_from_candidates = sum(ref.case_id in excluded_case_ids for ref in case_refs)
    if excluded_case_ids:
        case_refs = [ref for ref in case_refs if ref.case_id not in excluded_case_ids]
    if args.shuffle:
        rng = random.Random(args.sample_seed)
        rng.shuffle(case_refs)
    if args.n and not args.target_completed:
        case_refs = case_refs[: args.n]
    skipped_completed = 0
    skipped_failed = 0
    pending_refs = []
    for ref in case_refs:
        if ref.case_id in completed_case_ids:
            skipped_completed += 1
            continue
        if args.resume and not args.retry_failed and ref.case_id in failed_case_ids:
            skipped_failed += 1
            continue
        pending_refs.append(ref)

    patient_provider_name = args.patient_provider or args.env_provider or cfg["default_llm_provider"]
    lab_provider_name = args.lab_provider or patient_provider_name
    doctor_provider_name = args.doctor_provider or patient_provider_name
    patient_config = cfg["llm_providers"][patient_provider_name]
    lab_config = cfg["llm_providers"][lab_provider_name]
    doctor_config = cfg["llm_providers"][doctor_provider_name]
    max_turns = args.max_turns or cfg["simulation"]["max_turns"]
    memrl_input_path = resolve_memrl_input_path(args)
    memrl_bank = (
        load_memory_bank(memrl_input_path)
        if memrl_input_path or args.memrl_add_new or args.memrl_update_output
        else None
    )
    medagent_zero_input_path = resolve_medagent_zero_input_path(args)
    medagent_zero_bank = (
        medagent_zero_load_bank(medagent_zero_input_path)
        if medagent_zero_input_path or args.medagent_zero_add_new or args.medagent_zero_update_output
        else None
    )
    skillrl_memory = load_skillrl_memory(args)
    trace2skill_context = load_trace2skill_context(args.trace2skill_skill)

    print(f"Run ID: {run_id}")
    print(f"Cases: {len(case_refs)} (source={args.source or 'all'})")
    print(
        f"Patient provider: {patient_provider_name} | "
        f"Lab provider: {lab_provider_name} | "
        f"Doctor provider: {doctor_provider_name}"
    )
    print(f"Max turns: {max_turns}")
    if skillrl_memory is not None:
        print(f"SkillRL: bank={args.skillrl_bank} mode={args.skillrl_retrieval_mode}")
    if trace2skill_context is not None:
        print(f"Trace2Skill: skill={args.trace2skill_skill}")
    if exclusion_sources:
        print(
            f"Excluded {excluded_from_candidates} candidate cases from "
            f"{len(exclusion_sources)} rollout source(s)"
        )
    if args.resume:
        print(f"Resume: completed={len(existing_records)} failed={len(existing_errors)}")

    records: list[dict] = list(existing_records)
    errors: list[dict] = list(existing_errors)
    new_records = 0
    new_errors = 0
    score_mode = "a" if args.resume else "w"
    rollout_mode = "a" if args.resume else "w"
    error_mode = "a" if args.resume else "w"
    progress_total = len(pending_refs)
    progress_unit = "case"
    if args.target_completed and args.n is not None:
        progress_total = max(args.n - len(records), 0)
        progress_unit = "success"

    with open(scores_path, score_mode) as scores_file, open(rollouts_path, rollout_mode) as rollouts_file, open(errors_jsonl_path, error_mode) as errors_file:
        with tqdm(total=progress_total, desc="simulating", unit=progress_unit) as progress:
            for ref in pending_refs:
                if args.target_completed and args.n is not None and len(records) >= args.n:
                    break
                case_id = ref.case_id
                selected_memories = []
                selected_skillrl = None
                try:
                    case = None
                    memory_context = None
                    context_blocks = []
                    if memrl_bank is not None and args.memrl_top_k > 0:
                        case = case or json.loads(ref.path.read_text(encoding="utf-8"))
                        selected_memories = retrieve_memories(
                            memrl_bank,
                            intent_from_case(case),
                            top_k=args.memrl_top_k,
                            min_similarity=args.memrl_min_similarity,
                            weight_similarity=args.memrl_weight_similarity,
                            weight_q=args.memrl_weight_q,
                            q_min_threshold=args.memrl_q_min_threshold,
                        )
                        memory_context = format_memory_context(
                            selected_memories,
                            language=cfg["simulation"].get("language", "en"),
                        )
                        if memory_context:
                            context_blocks.append(memory_context)
                    if skillrl_memory is not None:
                        case = case or json.loads(ref.path.read_text(encoding="utf-8"))
                        selected_skillrl = skillrl_memory.retrieve(
                            skillrl_query_from_case(case),
                            top_k=args.skillrl_top_k,
                        )
                        skill_context = skillrl_memory.format_for_prompt(selected_skillrl)
                        if skill_context:
                            context_blocks.append(skill_context)
                    if trace2skill_context is not None:
                        context_blocks.append(trace2skill_context)
                    selected_medagent_zero = {"record_library": [], "experience_base": []}
                    if medagent_zero_bank is not None and (
                        args.medagent_zero_top_k_records > 0 or args.medagent_zero_top_k_experiences > 0
                    ):
                        case = case or json.loads(ref.path.read_text(encoding="utf-8"))
                        selected_medagent_zero = medagent_zero_retrieve(
                            medagent_zero_bank,
                            medagent_zero_query_from_case(case, args.medagent_zero_stage),
                            stage=args.medagent_zero_stage,
                            top_k_records=args.medagent_zero_top_k_records,
                            top_k_experiences=args.medagent_zero_top_k_experiences,
                            min_similarity=args.medagent_zero_min_similarity,
                        )
                        mz_context = medagent_zero_format_context(
                            selected_medagent_zero,
                            language=cfg["simulation"].get("language", "en"),
                        )
                        if mz_context:
                            context_blocks.append(mz_context)
                    memory_context = "\n\n".join(context_blocks) if context_blocks else None
                    session = SimulationSession.from_case_file(
                        case_path=ref.path,
                        patient_openai_config=patient_config,
                        lab_openai_config=lab_config,
                        datasets_config=datasets_config,
                        max_turns=max_turns,
                        language=cfg["simulation"].get("language", "en"),
                        doctor_openai_config=doctor_config,
                        doctor_memory_context=memory_context,
                    )
                    result = session.run()
                    # Persist evaluation results at a consistent precision while
                    # leaving the simulation's non-score trace data untouched.
                    result["scores"] = round_scores(result["scores"])
                except Exception as e:
                    error = {"case_id": case_id, "case_path": str(ref.path), "error": str(e)}
                    errors.append(error)
                    failed_case_ids.add(case_id)
                    new_errors += 1
                    errors_file.write(json.dumps(error, ensure_ascii=False) + "\n")
                    errors_file.flush()
                    progress.set_postfix(errors=new_errors)
                    print(f"[error] {case_id}: {e}", file=sys.stderr, flush=True)
                    continue

                trace_path = traces_dir / f"{case_id}.json"
                with open(trace_path, "w") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

                score_line = {"case_id": case_id, **result["scores"]}
                scores_file.write(json.dumps(score_line, ensure_ascii=False) + "\n")
                scores_file.flush()

                record = build_rollout_record(
                    case_path=ref.path,
                    result=result,
                    patient_provider=patient_provider_name,
                    doctor_provider=doctor_provider_name,
                    lab_provider=lab_provider_name,
                )
                record["scores"] = round_scores(record["scores"])
                if selected_memories:
                    reward = reward_from_scores(record["scores"])
                    selected_ids = [item["memory_id"] for item in selected_memories]
                    record["memrl"] = {
                        "selected_memory_ids": selected_ids,
                        "reward": reward,
                        "q_updates": update_memory_values(
                            memrl_bank,
                            selected_ids,
                            reward,
                            alpha=args.memrl_alpha,
                        ),
                    }
                if selected_skillrl is not None:
                    record["skillrl"] = {
                        "task_type": selected_skillrl.get("task_type"),
                        "retrieval_mode": selected_skillrl.get("retrieval_mode"),
                        "general_skill_ids": [item.get("skill_id") for item in selected_skillrl.get("general_skills", [])],
                        "task_skill_ids": [item.get("skill_id") for item in selected_skillrl.get("task_specific_skills", [])],
                    }
                if selected_medagent_zero["record_library"] or selected_medagent_zero["experience_base"]:
                    record["medagent_zero"] = {
                        "stage": args.medagent_zero_stage,
                        "selected_records": [
                            {"id": item.get("id"), "retrieval": item.get("retrieval", {})}
                            for item in selected_medagent_zero["record_library"]
                        ],
                        "selected_experiences": [
                            {"id": item.get("id"), "retrieval": item.get("retrieval", {})}
                            for item in selected_medagent_zero["experience_base"]
                        ],
                    }
                if args.memrl_add_new:
                    add_or_replace_memory(memrl_bank, record)
                if medagent_zero_bank is not None and args.medagent_zero_add_new:
                    medagent_zero_add_or_replace_record(medagent_zero_bank, record)
                records.append(record)
                completed_case_ids.add(case_id)
                new_records += 1
                rollouts_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                rollouts_file.flush()
                progress.update(1)
                progress.set_postfix(errors=new_errors)

    summary = generate_report(traces_dir, results_dir / "summary.json")
    metadata = {
        "run_id": run_id,
        "cases_dir": str(cases_dir.resolve()),
        "source": args.source,
        "n_requested": args.n,
        "n_completed": len(records),
        "n_failed": len(errors),
        "n_new_completed": new_records,
        "n_new_failed": new_errors,
        "shuffle": args.shuffle,
        "sample_seed": args.sample_seed,
        "exclude_case_ids_from": exclusion_sources,
        "excluded_case_ids": len(excluded_case_ids),
        "excluded_from_candidate_cases": excluded_from_candidates,
        "resume": args.resume,
        "retry_failed": args.retry_failed,
        "target_completed": args.target_completed,
        "target_met": args.n is None or not args.target_completed or len(records) >= args.n,
        "skipped_completed_cases": skipped_completed,
        "skipped_failed_cases": skipped_failed,
        "pending_cases_at_start": len(pending_refs),
        "patient_provider": patient_provider_name,
        "lab_provider": lab_provider_name,
        "doctor_provider": doctor_provider_name,
        "max_turns": max_turns,
        "skillrl_bank": args.skillrl_bank,
        "skillrl_top_k": args.skillrl_top_k,
        "skillrl_task_specific_top_k": args.skillrl_task_specific_top_k,
        "skillrl_retrieval_mode": args.skillrl_retrieval_mode,
        "trace2skill_skill": args.trace2skill_skill,
    }
    (results_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (results_dir / "errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== RUN SUMMARY (n={summary['n']}) ===")
    metrics = summary.get("metrics", {})
    for label, metric_key in SUMMARY_METRICS:
        stats = metrics.get(metric_key)
        if not stats:
            continue
        print(f"  {label}: mean={stats['mean']:.4f} ± {stats['std']:.4f}")
    if memrl_bank is not None and args.memrl_update_output:
        save_memory_bank(memrl_bank, args.memrl_update_output)
    if medagent_zero_bank is not None and args.medagent_zero_update_output:
        medagent_zero_save_bank(medagent_zero_bank, args.medagent_zero_update_output)
    print(f"\nResults saved to {results_dir}")
    if errors and not records:
        raise SystemExit(1)
    if args.target_completed and args.n is not None and len(records) < args.n:
        raise SystemExit(
            f"Only completed {len(records)} of the requested {args.n} successful evaluations; "
            "the eligible case pool was exhausted."
        )


if __name__ == "__main__":
    main()
