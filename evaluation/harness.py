"""Shared agent evaluation and training entry points."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import random
import statistics
import uuid
from pathlib import Path
from core.config import load_yaml_config
from env.clinical_env import ClinicalEnv
from env.runner import run_episode
from env.task_stream import TaskStream
from doctor_train.rollouts import build_rollout_record
from evaluation.reporter import generate_report
from examples.memory_trainer import MemoryTrainer

ROOT = Path(__file__).resolve().parents[1]


def import_object(name: str):
    module, attribute = name.split(':', 1)
    return getattr(importlib.import_module(module), attribute)


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def load_case_ids(paths: list[str]) -> set[str]:
    ids = set()
    for name in paths:
        path = Path(name)
        if path.is_dir():
            path = path / 'rollouts.jsonl'
        if path.suffix == '.jsonl':
            records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            ids.update(record['case_id'] for record in records)
        else:
            ids.update(line.strip() for line in path.read_text().splitlines() if line.strip())
    return ids


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description='Evaluate or train clinical agents')
    result.add_argument('--config', default='config/default.yaml')
    result.add_argument('--agent', default='examples.openai_agent:OpenAIAgent')
    result.add_argument('--cases-dir')
    result.add_argument('--case-ids-file', '--include-case-ids-from', dest='include', action='append', default=[])
    result.add_argument('--exclude-case-ids-from', dest='exclude', action='append', default=[])
    result.add_argument('--n', type=int)
    result.add_argument('--source')
    result.add_argument('--shuffle', action='store_true')
    result.add_argument('--sample-seed', type=int, default=2026)
    result.add_argument('--output', '--output-root', dest='output')
    result.add_argument('--run-id')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--retry-failed', action='store_true')
    result.add_argument('--fail-fast', action='store_true')
    result.add_argument('--repeats', type=int, default=1)
    result.add_argument('--max-turns', type=int)
    result.add_argument('--patient-provider')
    result.add_argument('--doctor-provider')
    result.add_argument('--lab-provider')
    result.add_argument('--disable-clinlab', action='store_true')
    result.add_argument('--lab-mask-ratio', type=float, default=0.0)
    result.add_argument('--lab-mask-seed', type=int, default=0)
    result.add_argument('--trainer')
    result.add_argument('--trainer-state')
    result.add_argument('--memory-state', '--memrl-memory', dest='memory_state')
    result.add_argument('--memrl-top-k', type=int, default=3)
    result.add_argument('--memrl-recall-k', type=int, default=20)
    result.add_argument('--memrl-alpha', type=float, default=0.1)
    result.add_argument('--memrl-update-output')
    return result


def validate(args) -> None:
    if args.n is not None and args.n < 1 or args.max_turns is not None and args.max_turns < 1:
        raise ValueError('Case count and turn budget must be positive')
    if args.repeats < 1 or not 0 <= args.lab_mask_ratio <= 1:
        raise ValueError('Invalid repeat count or laboratory mask ratio')
    if args.disable_clinlab and args.lab_mask_ratio:
        raise ValueError('Lab masking requires ClinLab')
    if args.trainer and (not args.trainer_state or not args.include):
        raise ValueError('Training requires a state path and training manifest')
    if args.trainer and (args.memory_state or args.memrl_update_output):
        raise ValueError('Select a trainer or a MemRL state')
    if args.memrl_update_output and not args.include:
        raise ValueError('Memory updates require a training manifest')
    if args.resume and (args.trainer or args.memrl_update_output):
        raise ValueError('Start a new output directory when updating training state')
    if args.repeats > 1 and (args.trainer or args.memrl_update_output):
        raise ValueError('Repeated evaluation requires a frozen state')
    if not 1 <= args.memrl_top_k <= args.memrl_recall_k or not 0 < args.memrl_alpha <= 1:
        raise ValueError('Invalid MemRL selection sizes or learning rate')


def evaluate(args, cfg: dict, output: Path, seed: int) -> dict:
    refs = list(TaskStream(args.cases_dir or ROOT / cfg['paths']['cases'], source=args.source))
    included, excluded = load_case_ids(args.include), load_case_ids(args.exclude)
    if included - {ref.case_id for ref in refs}:
        raise ValueError('The manifest contains missing or unusable cases')
    refs = [ref for ref in refs if (not included or ref.case_id in included) and ref.case_id not in excluded]
    if args.shuffle:
        random.Random(seed).shuffle(refs)
    if args.n:
        refs = refs[:args.n]
    if not refs:
        raise ValueError('No usable cases selected')
    if args.n and len(refs) < args.n:
        raise ValueError('The selected collection is smaller than the requested case count')
    providers = {role: getattr(args, f'{role}_provider') or cfg[f'default_{role}_provider'] for role in ('patient', 'doctor', 'lab')}
    identity = {'agent': args.agent, 'case_paths': [str(ref.path.resolve()) for ref in refs],
                'config': cfg, 'seed': seed, 'providers': providers, 'max_turns': args.max_turns,
                'lab_mask_ratio': args.lab_mask_ratio, 'lab_mask_seed': args.lab_mask_seed,
                'disable_clinlab': args.disable_clinlab, 'memory_state': args.memory_state,
                'memory_hash': hashlib.sha256(Path(args.memory_state).read_bytes()).hexdigest() if args.memory_state else None,
                'n': args.n, 'memrl_top_k': args.memrl_top_k, 'memrl_recall_k': args.memrl_recall_k}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError('Choose a new output directory or use --resume')
    traces = output / 'traces'
    if args.resume:
        metadata_path = output / 'metadata.json'
        if not metadata_path.exists() or json.loads(metadata_path.read_text()).get('fingerprint') != fingerprint:
            raise ValueError('Resume settings do not match the saved run')
    traces.mkdir(parents=True, exist_ok=True)
    metadata = {'fingerprint': fingerprint, 'agent': args.agent, 'providers': providers,
                'training': bool(args.trainer or args.memrl_update_output), 'cases': len(refs), 'n_completed': 0}
    write_json(output / 'metadata.json', metadata)
    selected_ids = {ref.case_id for ref in refs}
    records = [json.loads(path.read_text()) for path in sorted(traces.glob('*.json'))]
    if any(record['case_id'] not in selected_ids for record in records):
        raise ValueError('Output contains traces outside the selected collection')
    completed = {record['case_id'] for record in records}
    if len(completed) != len(records):
        raise ValueError('Output contains duplicate cases')
    errors_path = output / 'errors.json'
    errors = json.loads(errors_path.read_text()) if args.resume and errors_path.exists() else []
    failed = {error['case_id'] for error in errors}
    for filename, values in [('rollouts.jsonl', records), ('scores.jsonl', [{'case_id': r['case_id'], **r['scores']} for r in records])]:
        (output / filename).write_text(''.join(json.dumps(value, ensure_ascii=False) + '\n' for value in values))
    trainer = None
    if args.trainer:
        state = args.trainer_state if Path(args.trainer_state).exists() else None
        trainer = import_object(args.trainer)(state, config=cfg)
    elif args.memrl_update_output:
        state = args.memory_state or (args.memrl_update_output if Path(args.memrl_update_output).exists() else None)
        trainer = MemoryTrainer(state, config=cfg, top_k=args.memrl_top_k, recall_k=args.memrl_recall_k, alpha=args.memrl_alpha)
    memory = trainer or (MemoryTrainer(args.memory_state, config=cfg, top_k=args.memrl_top_k, recall_k=args.memrl_recall_k) if args.memory_state else None)
    datasets = {name: {**entry, 'data_dir': str((ROOT / entry['data_dir']).resolve())} for name, entry in cfg['datasets'].items()}
    agent_factory = import_object(args.agent)
    for ref in refs:
        if ref.case_id in completed or ref.case_id in failed and not args.retry_failed:
            continue
        if args.n and len(records) >= args.n:
            break
        try:
            env = ClinicalEnv.from_case_file(ref.path, cfg['llm_providers'][providers['patient']],
                lab_openai_config=cfg['llm_providers'][providers['lab']], datasets_config=datasets,
                max_turns=args.max_turns or cfg['simulation']['max_turns'], language=cfg['simulation'].get('language', 'en'),
                clinlab_enabled=not args.disable_clinlab, lab_mask_ratio=args.lab_mask_ratio, lab_mask_seed=args.lab_mask_seed)
            agent = agent_factory(cfg['llm_providers'][providers['doctor']], cfg['simulation'].get('language', 'en'))
            result = run_episode(env, agent, context_provider=memory.context if memory else None)
            record = build_rollout_record(case_path=ref.path, result=result, patient_provider=providers['patient'],
                doctor_provider=args.agent, lab_provider=providers['lab'], split='train' if trainer else 'evaluation')
            if isinstance(memory, MemoryTrainer):
                record['memrl'] = {'selected_memory_ids': list(memory.selected_ids), 'updated': bool(trainer)}
            elif trainer:
                record['training'] = {'trainer': args.trainer}
            if trainer:
                trainer.update(record)
                trainer.save(args.trainer_state or args.memrl_update_output)
            write_json(traces / f'{ref.case_id}.json', record)
            records.append(record)
            completed.add(ref.case_id)
            errors = [error for error in errors if error['case_id'] != ref.case_id]
            for filename, value in [('rollouts.jsonl', record), ('scores.jsonl', {'case_id': ref.case_id, **record['scores']})]:
                with (output / filename).open('a') as handle:
                    handle.write(json.dumps(value, ensure_ascii=False) + '\n')
        except Exception as error:
            errors = [e for e in errors if e['case_id'] != ref.case_id] + [{'case_id': ref.case_id, 'error': f'{type(error).__name__}: {error}'}]
            write_json(errors_path, errors)
            if args.fail_fast or trainer:
                raise
    write_json(errors_path, errors)
    summary = generate_report(traces, output / 'summary.json')
    metadata['n_completed'] = len(records)
    metadata['n_failed'] = len(errors)
    write_json(output / 'metadata.json', metadata)
    if errors or args.n and len(records) < args.n:
        raise RuntimeError(f"Completed {len(records)} cases with {len(errors)} errors; inspect {errors_path}")
    print(f'Completed {len(records)} cases: {output}')
    return summary


def main(argv=None):
    args = parser().parse_args(argv)
    validate(args)
    cfg = load_yaml_config(args.config)
    output = Path(args.output) if args.output else ROOT / cfg['paths']['eval_results'] / (args.run_id or uuid.uuid4().hex[:8])
    if args.repeats == 1:
        return evaluate(args, cfg, output, args.sample_seed)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError('Choose a new output directory or use --resume')
    output.mkdir(parents=True, exist_ok=True)
    summaries = [evaluate(args, cfg, output / f'repeat_{i + 1:02d}', args.sample_seed + i) for i in range(args.repeats)]
    metrics = {}
    for key in summaries[0].get('metrics', {}):
        means = [summary['metrics'][key]['mean'] for summary in summaries]
        metrics[key] = {'mean': statistics.mean(means), 'std': statistics.stdev(means), 'repeat_means': means}
    result = {'n': sum(summary['n'] for summary in summaries), 'repeats': args.repeats, 'metrics': metrics}
    write_json(output / 'summary.json', result)
    return result
