# Agent evaluation

## Agent protocol

`agents/interfaces.py` defines `EpisodeContext`, `Action` and `DoctorAgent`.

| Method | Input or output |
|---|---|
| `reset(context)` | Initial presentation, tool schemas, turn budget and optional memory |
| `act()` | `Action(name, arguments)` |
| `observe(text)` | Text returned by the environment |

Available actions are supplied in `context.tools_schema`: patient inquiry, physical examination, laboratory testing, imaging when supported, and final diagnosis.

`examples/openai_agent.py` implements this protocol using `GPTDoctorAgent`. A custom agent class takes `(provider, language)` at construction. Agents receive observable state and tool results; completed scores and trajectories are handled by the harness.

## Reference agent

```bash
python scripts/run_agent.py \
  --agent examples.openai_agent:OpenAIAgent \
  --case-ids-file data/splits/test.txt \
  --n 100 \
  --output outputs/evaluation/reference
```

## Custom agent

```bash
python scripts/run_agent.py \
  --agent my_package.my_agent:MyAgent \
  --case-ids-file data/splits/test.txt \
  --n 100 \
  --output outputs/evaluation/custom
```

The package containing `MyAgent` must be importable in the active environment.

## Environment loop

```python
from env.clinical_env import ClinicalEnv
from env.runner import run_episode
from examples.openai_agent import OpenAIAgent
from core.config import load_yaml_config

cfg = load_yaml_config("config/local.yaml")
env = ClinicalEnv.from_case_file(
    case_path,
    cfg["llm_providers"]["patient"],
    lab_openai_config=cfg["llm_providers"]["lab"],
    datasets_config=cfg["datasets"],
)
agent = OpenAIAgent(cfg["llm_providers"]["doctor"])
result = run_episode(env, agent)
```

## Batch model benchmark

```bash
python scripts/run_batch_eval.py \
  --cases-dir data/cases \
  --include-case-ids-from data/splits/test.txt \
  --patient-provider patient \
  --doctor-provider doctor \
  --lab-provider lab \
  --n 100 \
  --run-id reference \
  --output-root outputs/benchmark/reference
```

Change the `doctor` endpoint/model in `.env` to evaluate another backbone. `--repeats 5` runs five evaluations. `--disable-clinlab` selects the corresponding environment condition.

## Metrics

| Metric | Definition |
|---|---|
| ICR ↑ | Weighted coverage of initially hidden, patient-reportable information |
| TOC ↑ | Coverage of case-supported examination findings |
| ATC ↑ | Coverage of abnormal laboratory and microbiology findings |
| DCA ↑ | ICD-10 block agreement |
| DSA ↑ | Three-character ICD-10 category agreement |

A run writes `rollouts.jsonl`, `scores.jsonl`, `traces/`, `metadata.json`, `errors.json` and `summary.json`.

## HTTP interface

```bash
python scripts/run_api_server.py --config config/local.yaml --port 8080
```

| Endpoint | Purpose |
|---|---|
| `GET /cases` | List case IDs |
| `POST /sessions` | Start with `{"case_id": "..."}` |
| `POST /sessions/{id}/step` | Send `{"tool": "ask_patient", "args": {"question": "..."}}` |
| `GET /sessions/{id}/score` | Retrieve scores after completion |
| `DELETE /sessions/{id}` | Close a session |

Interactive API documentation is available at `/docs`.

## Run management

Use a new output directory for each experiment. `--resume --retry-failed` continues the same evaluation settings and retries failed episodes. Failed model requests are recorded in `errors.json`; an incomplete run exits with an error. Resume checks the configuration and memory-state fingerprint.
