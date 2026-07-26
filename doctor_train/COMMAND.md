# Doctor-Agent Rollout and Training Commands

Run the commands below from the ClinVerse repository root. Configure the
required model endpoints in `config/default.yaml` and export their API keys
before starting.

These integrations reproduce method-level data structures and retrieval
strategies. They are not official reproductions of the upstream projects.

## Collect Rollouts

Collect 100 normalized trajectories:

```bash
python scripts/run_doctor_rollouts.py \
  --source mimic-iv-ext \
  --n 100 \
  --patient-provider gpt-4o-mini \
  --doctor-provider gpt-4o-mini \
  --run-id train100
```

By default, this writes:

```text
data/doctor_train/runs/train100/
├── metadata.json
├── summary.json
├── errors.json
├── errors.jsonl
├── scores.jsonl
├── rollouts.jsonl
└── traces/
```

Use independent providers for the environment roles and evaluated doctor:

```bash
python scripts/run_doctor_rollouts.py \
  --source mimic-iv-ext \
  --n 100 \
  --patient-provider qwen3-32b-vllm \
  --lab-provider Qwen3.5-2B-mimic-iv-ext \
  --doctor-provider gpt-4o-mini \
  --max-turns 20 \
  --run-id train100_mixed
```

Resume an interrupted run and retry failed cases:

```bash
python scripts/run_doctor_rollouts.py \
  --source mimic-iv-ext \
  --n 100 \
  --run-id train100 \
  --resume \
  --retry-failed \
  --target-completed
```

Useful sampling controls:

```bash
# Reproducible random sampling
python scripts/run_doctor_rollouts.py \
  --source mimic-iv-ext --n 100 --sample-seed 2026 --run-id seeded100

# Preserve case-index order
python scripts/run_doctor_rollouts.py \
  --source mimic-iv-ext --n 100 --no-shuffle --run-id ordered100
```

## Trace2Skill

Export both successful and failed trajectories into Trace2Skill-style
analysis records:

```bash
python scripts/export_trace2skill_data.py \
  --input-jsonl data/doctor_train/runs/train100/rollouts.jsonl \
  --output-dir data/doctor_train/runs/train100/trace2skill \
  --mode combined
```

Export only failure analysis:

```bash
python scripts/export_trace2skill_data.py \
  --input-jsonl data/doctor_train/runs/train100/rollouts.jsonl \
  --output-dir data/doctor_train/runs/train100/trace2skill_errors \
  --mode error \
  --diagnosis-threshold 1.0
```

Use the bundled static clinical skill during evaluation:

```bash
python scripts/run_batch_eval.py \
  --source mimic-iv-ext \
  --n 100 \
  --trace2skill-skill doctor_train/trace2skill_base/clinical/SKILL.md \
  --run-id trace2skill_eval
```

## SkillRL

Distill rollout trajectories into a SkillRL-compatible skill bank:

```bash
python scripts/build_skillrl_skillbank.py \
  --input-jsonl data/doctor_train/runs/train100/rollouts.jsonl \
  --output data/doctor_train/skillrl_bank.json \
  --distiller-provider gpt-4o-mini
```

Evaluate with template-based skill retrieval:

```bash
python scripts/run_batch_eval.py \
  --source mimic-iv-ext \
  --n 100 \
  --skillrl-bank data/doctor_train/skillrl_bank.json \
  --skillrl-top-k 6 \
  --skillrl-task-specific-top-k 3 \
  --run-id skillrl_eval
```

The public repository supports template retrieval directly. The optional
`validate_skillrl_skillbank.py` compatibility command requires a separate
checkout of the upstream SkillRL repository.

## MemRL

Build a MemRL-style episodic memory bank:

```bash
python scripts/build_memrl_memory.py \
  --input-jsonl data/doctor_train/runs/train100/rollouts.jsonl \
  --output data/doctor_train/memrl_memory.json
```

Evaluate with the frozen memory bank:

```bash
python scripts/run_batch_eval.py \
  --source mimic-iv-ext \
  --n 100 \
  --memrl-memory data/doctor_train/memrl_memory.json \
  --memrl-top-k 3 \
  --run-id memrl_offline_eval
```

Run online memory updates while collecting new trajectories:

```bash
python scripts/run_doctor_rollouts.py \
  --source mimic-iv-ext \
  --n 100 \
  --memrl-memory data/doctor_train/memrl_memory.json \
  --memrl-top-k 3 \
  --memrl-update-output data/doctor_train/memrl_memory.updated.json \
  --memrl-add-new \
  --run-id memrl_online
```

Retrieval behavior can be adjusted with
`--memrl-min-similarity`, `--memrl-alpha`,
`--memrl-weight-similarity`, `--memrl-weight-q`, and
`--memrl-q-min-threshold`.

## MedAgent-Zero

Build the stage-aware record and experience libraries:

```bash
python scripts/build_medagent_zero_bank.py \
  --input-jsonl data/doctor_train/runs/train100/rollouts.jsonl \
  --output data/doctor_train/medagent_zero_bank.json
```

Evaluate with both successful records and failure-derived experiences:

```bash
python scripts/run_batch_eval.py \
  --source mimic-iv-ext \
  --n 100 \
  --medagent-zero-bank data/doctor_train/medagent_zero_bank.json \
  --medagent-zero-stage diagnosis \
  --medagent-zero-top-k-records 3 \
  --medagent-zero-top-k-experiences 3 \
  --run-id medagent_zero_combined
```

Ablation settings:

```bash
# Record library only
python scripts/run_batch_eval.py \
  --source mimic-iv-ext \
  --medagent-zero-bank data/doctor_train/medagent_zero_bank.json \
  --medagent-zero-stage diagnosis \
  --medagent-zero-top-k-records 3 \
  --medagent-zero-top-k-experiences 0

# Experience base only
python scripts/run_batch_eval.py \
  --source mimic-iv-ext \
  --medagent-zero-bank data/doctor_train/medagent_zero_bank.json \
  --medagent-zero-stage diagnosis \
  --medagent-zero-top-k-records 0 \
  --medagent-zero-top-k-experiences 3
```

Update the two libraries during rollout collection:

```bash
python scripts/run_doctor_rollouts.py \
  --source mimic-iv-ext \
  --n 100 \
  --medagent-zero-bank data/doctor_train/medagent_zero_bank.json \
  --medagent-zero-stage diagnosis \
  --medagent-zero-top-k-records 3 \
  --medagent-zero-top-k-experiences 3 \
  --medagent-zero-update-output data/doctor_train/medagent_zero_bank.updated.json \
  --medagent-zero-add-new \
  --run-id medagent_zero_online
```

## Command Reference

Use the built-in help for the complete, current argument list:

```bash
python scripts/run_doctor_rollouts.py --help
python scripts/run_batch_eval.py --help
python scripts/export_trace2skill_data.py --help
python scripts/build_skillrl_skillbank.py --help
python scripts/build_memrl_memory.py --help
python scripts/build_medagent_zero_bank.py --help
```
