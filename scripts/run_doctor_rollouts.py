#!/usr/bin/env python3
"""Collect normalized doctor-agent rollout records for training baselines."""

from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from core.config import load_yaml_config
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
from doctor_train.rollouts import build_rollout_record, load_case, summarize_rollouts
from env.task_stream import TaskStream
from simulation.session import SimulationSession


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


def memory_case_ids(bank: dict | None) -> set[str]:
    if bank is None:
        return set()
    return {
        str(item.get("metadata", {}).get("case_id"))
        for item in bank.get("items", [])
        if item.get("metadata", {}).get("case_id")
    }


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


def load_existing_errors(output_root: Path) -> list[dict]:
    errors_jsonl = output_root / "errors.jsonl"
    if errors_jsonl.exists():
        return load_jsonl_records(errors_jsonl)
    errors_json = output_root / "errors.json"
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


def medagent_zero_input_path(args) -> str | None:
    if args.medagent_zero_bank:
        return args.medagent_zero_bank
    if args.medagent_zero_update_output and Path(args.medagent_zero_update_output).exists():
        return args.medagent_zero_update_output
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run doctor-agent rollouts and store normalized training records.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--cases-dir", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=None, help="Random seed for case sampling. Omit for a fresh random sample each run.")
    parser.add_argument("--no-shuffle", action="store_true", help="Keep index order instead of sampling cases randomly.")
    parser.add_argument("--allow-repeat-cases", action="store_true", help="Do not exclude cases already present in the loaded MemRL bank.")
    parser.add_argument("--resume", action="store_true", help="Append to an existing output root and skip completed/failed cases.")
    parser.add_argument("--retry-failed", action="store_true", help="With --resume, retry cases already present in errors.json/errors.jsonl.")
    parser.add_argument("--target-completed", action="store_true", help="Treat --n as the target number of successful rollouts, replacing failed cases with later sampled cases.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--patient-provider", default=None)
    parser.add_argument("--lab-provider", default=None)
    parser.add_argument("--doctor-provider", default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--only-usable", action="store_true", default=True)
    parser.add_argument("--memrl-memory", default=None, help="Hospital-MemRL memory bank JSON to retrieve from.")
    parser.add_argument("--memrl-top-k", type=int, default=0, help="Number of retrieved memories to inject into the doctor prompt.")
    parser.add_argument("--memrl-min-similarity", type=float, default=0.0)
    parser.add_argument("--memrl-alpha", type=float, default=0.1, help="Q-value update rate for selected memories.")
    parser.add_argument("--memrl-weight-similarity", type=float, default=0.5)
    parser.add_argument("--memrl-weight-q", type=float, default=0.5)
    parser.add_argument("--memrl-q-min-threshold", type=float, default=None)
    parser.add_argument("--memrl-update-output", default=None, help="Write updated memory bank here after the run.")
    parser.add_argument("--memrl-add-new", action="store_true", help="Add each completed rollout as a new memory.")
    parser.add_argument("--medagent-zero-bank", default=None, help="MedAgent-Zero dual-library JSON to retrieve from.")
    parser.add_argument("--medagent-zero-stage", default="diagnosis", choices=["examination", "diagnosis", "treatment"])
    parser.add_argument("--medagent-zero-top-k-records", type=int, default=0)
    parser.add_argument("--medagent-zero-top-k-experiences", type=int, default=0)
    parser.add_argument("--medagent-zero-min-similarity", type=float, default=0.0)
    parser.add_argument("--medagent-zero-update-output", default=None, help="Write updated MedAgent-Zero bank here after the run.")
    parser.add_argument("--medagent-zero-add-new", action="store_true", help="Add each completed rollout into the MedAgent-Zero dual library.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_root = Path(__file__).parent.parent
    config_path = Path(args.config)
    cases_dir = resolve_config_path(config_path, args.cases_dir or cfg["paths"]["cases"])
    run_id = args.run_id or uuid.uuid4().hex[:8]
    output_root = Path(args.output_root or repo_root / "data" / "doctor_train" / "runs" / run_id)
    traces_dir = output_root / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_root / "scores.jsonl"
    rollouts_path = output_root / "rollouts.jsonl"
    errors_jsonl_path = output_root / "errors.jsonl"

    existing_records = load_jsonl_records(rollouts_path) if args.resume else []
    existing_errors = load_existing_errors(output_root) if args.resume else []
    if args.resume:
        seed_errors_jsonl(errors_jsonl_path, existing_errors)
    completed_case_ids = {str(record.get("case_id")) for record in existing_records if record.get("case_id")}
    failed_case_ids = {str(error.get("case_id")) for error in existing_errors if error.get("case_id")}

    datasets_config = {
        name: {"data_dir": str((repo_root / cfg_entry["data_dir"]).resolve())}
        for name, cfg_entry in cfg.get("datasets", {}).items()
        if cfg_entry.get("data_dir")
    }

    patient_provider_name = args.patient_provider or cfg["default_llm_provider"]
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
    medagent_zero_path = medagent_zero_input_path(args)
    medagent_zero_bank = (
        medagent_zero_load_bank(medagent_zero_path)
        if medagent_zero_path or args.medagent_zero_add_new or args.medagent_zero_update_output
        else None
    )
    case_refs = list(TaskStream(cases_dir, only_usable=args.only_usable, source=args.source))
    excluded_case_ids = set()
    if memrl_input_path and args.memrl_add_new and not args.allow_repeat_cases:
        excluded_case_ids = memory_case_ids(memrl_bank)
        case_refs = [ref for ref in case_refs if ref.case_id not in excluded_case_ids]
    if not args.no_shuffle:
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

    records: list[dict] = list(existing_records)
    errors: list[dict] = list(existing_errors)
    new_records = 0
    new_errors = 0
    score_mode = "a" if args.resume else "w"
    rollout_mode = "a" if args.resume else "w"
    error_mode = "a" if args.resume else "w"
    target_remaining = None
    progress_total = len(pending_refs)
    progress_unit = "case"
    if args.target_completed and args.n is not None:
        target_remaining = max(args.n - len(records), 0)
        progress_total = target_remaining
        progress_unit = "success"

    with open(scores_path, score_mode) as scores_file, open(rollouts_path, rollout_mode) as rollouts_file, open(errors_jsonl_path, error_mode) as errors_file:
        with tqdm(total=progress_total, desc="doctor-rollouts", unit=progress_unit) as progress:
            for ref in pending_refs:
                if args.target_completed and args.n is not None and len(records) >= args.n:
                    break
                try:
                    selected_memories = []
                    selected_medagent_zero = {"record_library": [], "experience_base": []}
                    memory_context = None
                    context_blocks = []
                    if memrl_bank is not None and args.memrl_top_k > 0:
                        case = load_case(ref.path)
                        intent = intent_from_case(case)
                        selected_memories = retrieve_memories(
                            memrl_bank,
                            intent,
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
                    if medagent_zero_bank is not None and (
                        args.medagent_zero_top_k_records > 0 or args.medagent_zero_top_k_experiences > 0
                    ):
                        case = load_case(ref.path)
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
                        doctor_openai_config=doctor_config,
                        datasets_config=datasets_config,
                        max_turns=max_turns,
                        language=cfg["simulation"].get("language", "en"),
                        doctor_memory_context=memory_context,
                    )
                    result = session.run()
                    trace_path = traces_dir / f"{ref.case_id}.json"
                    with open(trace_path, "w") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)

                    record = build_rollout_record(
                        case_path=ref.path,
                        result=result,
                        patient_provider=patient_provider_name,
                        doctor_provider=doctor_provider_name,
                        lab_provider=lab_provider_name,
                        split=args.split,
                    )
                    if selected_memories:
                        selected_ids = [m["memory_id"] for m in selected_memories]
                        reward = reward_from_scores(record["scores"])
                        updates = update_memory_values(
                            memrl_bank,
                            selected_ids,
                            reward,
                            alpha=args.memrl_alpha,
                        ) if memrl_bank is not None else []
                        record["memrl"] = {
                            "selected_memory_ids": selected_ids,
                            "selected_memories": [
                                {
                                    "memory_id": m["memory_id"],
                                    "retrieval": m.get("retrieval", {}),
                                    "success": m.get("metadata", {}).get("success"),
                                }
                                for m in selected_memories
                            ],
                            "reward": reward,
                            "q_updates": updates,
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
                    if memrl_bank is not None and args.memrl_add_new:
                        add_or_replace_memory(memrl_bank, record)
                    if medagent_zero_bank is not None and args.medagent_zero_add_new:
                        medagent_zero_add_or_replace_record(medagent_zero_bank, record)
                    records.append(record)
                    completed_case_ids.add(ref.case_id)
                    new_records += 1

                    scores_file.write(json.dumps({"case_id": ref.case_id, **result["scores"]}, ensure_ascii=False) + "\n")
                    rollouts_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    scores_file.flush()
                    rollouts_file.flush()

                    progress.update(1)
                    progress.set_postfix(errors=new_errors)
                except Exception as exc:
                    error = {"case_id": ref.case_id, "case_path": str(ref.path), "error": str(exc)}
                    errors.append(error)
                    failed_case_ids.add(ref.case_id)
                    new_errors += 1
                    errors_file.write(json.dumps(error, ensure_ascii=False) + "\n")
                    errors_file.flush()
                    progress.set_postfix(errors=new_errors)
                    print(f"[error] {ref.case_id}: {exc}", file=sys.stderr, flush=True)

    metadata = {
        "run_id": run_id,
        "cases_dir": str(cases_dir.resolve()),
        "source": args.source,
        "split": args.split,
        "n_requested": args.n,
        "n_completed": len(records),
        "n_failed": len(errors),
        "n_new_completed": new_records,
        "n_new_failed": new_errors,
        "sample_seed": args.sample_seed,
        "shuffle": not args.no_shuffle,
        "resume": args.resume,
        "retry_failed": args.retry_failed,
        "target_completed": args.target_completed,
        "skipped_completed_cases": skipped_completed,
        "skipped_failed_cases": skipped_failed,
        "pending_cases_at_start": len(pending_refs),
        "allow_repeat_cases": args.allow_repeat_cases,
        "excluded_existing_memory_cases": len(excluded_case_ids),
        "patient_provider": patient_provider_name,
        "doctor_provider": doctor_provider_name,
        "lab_provider": lab_provider_name,
        "max_turns": max_turns,
        "memrl_memory": args.memrl_memory,
        "memrl_loaded_from": memrl_input_path,
        "memrl_top_k": args.memrl_top_k,
        "memrl_min_similarity": args.memrl_min_similarity,
        "memrl_alpha": args.memrl_alpha,
        "memrl_weight_similarity": args.memrl_weight_similarity,
        "memrl_weight_q": args.memrl_weight_q,
        "memrl_q_min_threshold": args.memrl_q_min_threshold,
        "memrl_update_output": args.memrl_update_output,
        "memrl_add_new": args.memrl_add_new,
        "medagent_zero_bank": args.medagent_zero_bank,
        "medagent_zero_loaded_from": medagent_zero_path,
        "medagent_zero_stage": args.medagent_zero_stage,
        "medagent_zero_top_k_records": args.medagent_zero_top_k_records,
        "medagent_zero_top_k_experiences": args.medagent_zero_top_k_experiences,
        "medagent_zero_min_similarity": args.medagent_zero_min_similarity,
        "medagent_zero_update_output": args.medagent_zero_update_output,
        "medagent_zero_add_new": args.medagent_zero_add_new,
    }
    summary = summarize_rollouts(records)

    (output_root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    if memrl_bank is not None and args.memrl_update_output:
        save_memory_bank(memrl_bank, args.memrl_update_output)
    if medagent_zero_bank is not None and args.medagent_zero_update_output:
        medagent_zero_save_bank(medagent_zero_bank, args.medagent_zero_update_output)

    print(json.dumps({"output_root": str(output_root), "summary": summary, "errors": len(errors)}, ensure_ascii=False, indent=2))
    if errors and not records:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
