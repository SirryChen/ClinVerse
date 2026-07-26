# ClinVerse: A Realistic Multi-Agent Diagnostic Environment for Training and Evaluating Medical Agents

ClinVerse is an interactive, real-data-grounded multi-agent environment for
training and evaluating medical diagnostic agents. Unlike static medical
question-answering benchmarks, which present most relevant evidence up front
and primarily score the final answer, ClinVerse evaluates the complete
diagnostic process: what an agent asks, which examinations it orders, what
evidence it uncovers, and whether it integrates that evidence into the correct
ICD-10 diagnosis.

ClinVerse is built on **ClinPatient**, a scalable diagnostic patient pool of
**150,667 cases** derived from MIMIC-IV, MIMIC-IV-Note, and SICdb. A doctor
agent begins with only a limited patient presentation and must progressively
collect hidden patient information and request examinations through
open-ended interaction. The resulting trajectory is scored with reproducible,
rule-based criteria and can then be reused as experience, memory, or
transferable skills for agent training.

> **Research use only.** ClinVerse is not a medical device and must not be directly
> used for clinical care, diagnosis, or treatment decisions without supervision.

![ClinVerse framework](assets/main.png)


## Main Contributions

- **A realistic interactive diagnostic environment.** ClinVerse supports
  free-form consultation among six specialized roles and evaluates both
  medical agents and human participants under the same interaction protocol.
- **ClinPatient: 150K real-data-grounded cases.** A unified, extensible
  pipeline transforms heterogeneous EHR sources into admission-level
  diagnostic cases while preserving retrievable clinical evidence.
- **Leakage-resistant patient simulation.** A dual-layer patient architecture
  separates patient-visible knowledge from the latent physiological state,
  while a self-check mechanism filters unsupported or privileged disclosure.
- **Flexible examination ordering.** When an ordered result is absent from
  the source EHR, the ClinLab physiological-state predictor estimates a
  plausible result instead of exposing missingness as a diagnostic shortcut.
- **Scalable rule-based evaluation.** Five process and outcome metrics measure
  information collection, examination coverage, abnormal-finding coverage,
  and ICD-10 diagnosis at two levels of granularity without an LLM judge or
  case-specific expert rules.
- **A shared evaluation and training loop.** Interaction trajectories can be
  distilled into experience, episodic memory, or reusable skills. The
  repository includes method-level integrations for Trace2Skill, SkillRL,
  MemRL, and MedAgent-Zero.


## Repository Layout

```text
ClinVerse/
├── adapters/              # MIMIC-IV-Ext and SICdb-Ext adapters
├── agents/                # Reference tool-calling doctor agent
├── core/                  # Schemas, ports, registry, and configuration
├── doctor_train/          # Rollout and non-parametric training integrations
├── env/                   # ClinicalEnv, TaskStream, and HTTP API
├── evaluation/            # Information-collection and diagnosis metrics
├── experiments/           # Analysis and plotting code (results excluded)
├── pipeline/              # Case construction and Synthea matching
├── preprocessing/         # Raw MIMIC-IV/SICdb normalization pipelines
├── providers/             # Patient, examination, lab, and moderator tools
├── scripts/               # Main command-line entry points
├── simulation/            # Reference end-to-end simulation loop
├── synthea_extensions/    # Conditioned-generation modules and Java changes
├── training/              # Missing-laboratory-result model training code
├── config/
├── assets/
├── pyproject.toml
└── requirements.txt
```


## Installation

Python 3.10 or later is required.

```bash
git clone https://github.com/SirryChen/ClinVerse.git
cd ClinVerse

conda create -n clinverse python=3.10 -y
conda activate clinverse
pip install -e .
```

Configure an OpenAI or OpenAI-compatible endpoint:

```bash
export CLINVERSE_API_KEY="your-api-key"
export CLINVERSE_BASE_URL="https://api.openai.com/v1"
```

`config/default.yaml` also contains examples for local vLLM servers. Model
names, endpoints, timeouts, and role assignments can be changed there.
Credentials should be supplied through environment variables and must not be
committed.

## Data Preparation

The datasets must be obtained separately and used under their own licenses
and data-use agreements. In particular, MIMIC-derived data must never be
committed to this repository.

- [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/)
- [MIMIC-IV-Note v2.2](https://physionet.org/content/mimic-iv-note/2.2/)
- [SICdb v1.0.8](https://physionet.org/content/sicdb/1.0.8/)
- [Synthea](https://github.com/synthetichealth/synthea)

The default configuration expects datasets and Synthea beside the repository:

```text
workspace/
├── ClinVerse/
├── dataset/
│   ├── mimic-iv-3.1/          # credentialed source data
│   ├── mimic-iv-note/         # credentialed source notes
│   ├── SICdb/                 # source SICdb tables
│   ├── mimic-iv-ext/          # generated normalized adapter data
│   └── sicdb-ext/             # generated normalized adapter data
└── synthea/
    └── output_pool/
        └── fhir/
```

Expected adapter files include normalized demographics, history of present
illness, physical examination, laboratory results, diagnoses, medications,
and catalog files. Paths can be changed under `datasets` and `paths` in
`config/default.yaml`.

### Normalize source datasets

After placing credentialed raw data in the external `dataset/` directory,
build the normalized adapter inputs:

```bash
python preprocessing/mimic_iv_ext/predata/run.py
python preprocessing/mimic_iv_ext/predata/build_catalog.py

python preprocessing/sicdb_ext/predata/run.py
python preprocessing/sicdb_ext/predata/build_catalog.py
```

Use `--limit` with either `run.py` for a smaller pipeline smoke test. These
commands write only to the external dataset directories shown above.

### Synthea extensions

Clone Synthea next to ClinVerse, deploy the supplied modules and
conditioned-generation Java extensions, and build the JAR:

```bash
git clone https://github.com/synthetichealth/synthea.git ../synthea
./synthea_extensions/deploy.sh --synthea-root ../synthea

cd ../synthea
./gradlew uberJar
cd ../ClinVerse
```

Generated Synthea records belong in the external `synthea/` checkout and are
not part of this repository.

## Build Standardized Cases

Build cases by combining normalized EHR admissions with an existing Synthea
patient pool:

```bash
python scripts/build_cases.py \
  --dataset mimic-iv-ext \
  --limit 100 \
  --output data/cases
```

To run conditioned Synthea generation from dataset-derived priors:

```bash
python scripts/build_conditioned_cases.py \
  --dataset mimic-iv-ext \
  --limit 5 \
  --copies 10 \
  --output data/cases
```

Audit constructed cases for matching conflicts and leakage risk:

```bash
python scripts/audit_cases.py --cases-dir data/cases
```

All generated case files are ignored by Git.

## Run a Single Simulation

The reference simulation uses the configured doctor agent to control
`ClinicalEnv`:

```bash
python scripts/run_simulation.py \
  --case data/cases/CASE_ID.json \
  --output data/eval_results/single_case.json
```

The core in-process interface follows a Gym-like lifecycle:

```python
from env.clinical_env import ClinicalEnv

env = ClinicalEnv.from_case_file(
    "data/cases/CASE_ID.json",
    patient_openai_config=patient_config,
    datasets_config=datasets_config,
)

observation = env.reset()
step = env.step(
    "ask_patient",
    {"question": "When did your symptoms start?", "topics": ["onset"]},
)
step = env.step(
    "request_physical_exam",
    {"components": ["abdomen"], "reason": "Evaluate the abdominal complaint"},
)
step = env.step(
    "finalize_case",
    {
        "final_diagnosis": "Appendicitis",
        "icd10_code": "K35",
        "diagnosis_category": "appendicitis",
        "diagnosis_subtype": "acute appendicitis",
        "differential_diagnoses": ["gastroenteritis"],
        "supporting_evidence": ["Focused history and abdominal findings"],
    },
)
scores = env.score()
```

## HTTP API

For agent frameworks that cannot import the Python environment directly:

```bash
python scripts/run_api_server.py --host 0.0.0.0 --port 8000
```

The service exposes:

```text
GET    /cases?split=train|test|all
POST   /sessions
POST   /sessions/{session_id}/step
GET    /sessions/{session_id}/score
DELETE /sessions/{session_id}
```

Session state is stored in memory. The API is intended for research
experiments, not multi-worker production deployment.

## Batch Evaluation

Run a batch with independently selected patient, laboratory, and doctor
providers:

```bash
python scripts/run_batch_eval.py \
  --source mimic-iv-ext \
  --n 100 \
  --patient-provider gpt-4o-mini \
  --doctor-provider gpt-4o-mini \
  --run-id mimic100_gpt4omini
```

Useful controls include:

```bash
# Randomized sampling with a reproducible seed
python scripts/run_batch_eval.py \
  --source mimic-iv-ext --n 100 --shuffle --sample-seed 2026

# Resume an interrupted run
python scripts/run_batch_eval.py \
  --output-root data/eval_results/mimic100_gpt4omini \
  --resume --retry-failed
```

Each run writes per-case traces, errors, scores, metadata, and an aggregate
summary under `data/eval_results/`. These outputs are excluded from Git.

## Doctor-Agent Rollouts

Collect normalized trajectories for training or memory-based baselines:

```bash
python scripts/run_doctor_rollouts.py \
  --source mimic-iv-ext \
  --n 100 \
  --patient-provider gpt-4o-mini \
  --doctor-provider gpt-4o-mini \
  --run-id train100
```

The output includes `rollouts.jsonl`, `scores.jsonl`, summaries, errors, and
per-case traces.

For resume controls and the complete Trace2Skill, SkillRL, MemRL, and
MedAgent-Zero workflows, see
[doctor_train/COMMAND.md](doctor_train/COMMAND.md).

## Missing-Laboratory-Result Model

The provider-training code fine-tunes a local causal language model to predict
laboratory results when a requested result is absent from the case evidence.
Install its additional dependencies first:

```bash
cd training/provider/mimic_iv_ext
pip install -r requirements.txt

python prepare_data.py --config config.yaml
python train.py --config config.yaml
python evaluate.py \
  --config config.yaml \
  --adapter output/checkpoints/final_adapter

cd ../../..
```

The training configuration points to external model and dataset directories.
Prepared JSONL files, tokenizer caches, checkpoints, and predictions are all
ignored by Git.


## Acknowledgements

ClinPatient is derived from MIMIC-IV, MIMIC-IV-Note, and SICdb and uses
Synthea for controlled background augmentation. The training experiments
build on Trace2Skill, SkillRL, MemRL, and MedAgent-Zero. Please cite the
original datasets and methods when using the corresponding components.


## License

The code in this repository is released under the
[Apache License 2.0](LICENSE). External datasets, models, and upstream
projects remain subject to their own licenses and terms.
