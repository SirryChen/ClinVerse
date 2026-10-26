# Agent training

Use [ClinLab-2B](https://huggingface.co/SII-Sirry/ClinLab-2B) with the [local checkpoint setup](clinlab.md#use-pretrained-weights). Add `--config config/local.yaml` to the training and evaluation commands below.

## Trainer interface

```python
class Trainer:
    def __init__(self, state_path=None, *, config):
        self.config = config

    def context(self, episode):
        return self.retrieve(episode.visible_info)

    def update(self, record):
        self.optimize(record["transcript"], record["scores"])

    def save(self, path):
        self.save_checkpoint(path)
```

`context` receives an `EpisodeContext` containing the initial presentation, tools and turn budget. `update` receives a completed trajectory and feedback. `save` persists the training state. The executable implementation is [MemoryTrainer](../examples/memory_trainer.py).

## MemRL configuration

Configure the `embedding` provider in `config/default.yaml` or `.env`:

```bash
export CLINVERSE_EMBEDDING_BASE_URL=http://localhost:8002/v1
export CLINVERSE_EMBEDDING_API_KEY=EMPTY
export CLINVERSE_EMBEDDING_MODEL=BAAI/bge-m3
```

Each memory stores an intent, its embedding, a trajectory and a utility value. Retrieval recalls the 20 most similar intents by cosine similarity, then selects 3 memories by utility. `--memrl-recall-k` and `--memrl-top-k` control these sizes. The query and bank use the same embedding model.

The reward is `0.5 × diagnosis_score + 0.3 × ICR + 0.1 × TOC + 0.1 × ATC`. Online feedback updates retrieved utilities with `Q ← (1 − α)Q + αr`; the default α is 0.1.

## Offline training

Collect trajectories from the training split:

```bash
python scripts/run_batch_eval.py \
  --include-case-ids-from data/splits/train.txt \
  --n 1000 --output-root outputs/offline/episodes

python scripts/build_memrl_memory.py \
  --input-jsonl outputs/offline/episodes/rollouts.jsonl \
  --output outputs/memrl/memory.json
```

Evaluate on the test split:

```bash
python scripts/run_batch_eval.py \
  --include-case-ids-from data/splits/test.txt \
  --memrl-memory outputs/memrl/memory.json \
  --memrl-top-k 3 --memrl-recall-k 20 \
  --n 100 --output-root outputs/memrl/test
```

Evaluation leaves memory utilities and contents unchanged.

## Online training

```bash
python scripts/run_batch_eval.py \
  --include-case-ids-from data/splits/train.txt \
  --memrl-memory outputs/memrl/memory.json \
  --memrl-update-output outputs/memrl_online/memory.json \
  --memrl-alpha 0.1 \
  --n 1000 --output-root outputs/memrl_online/train
```

Each completed episode updates retrieved utilities and adds its experience. Omit `--memrl-memory` to start with an empty bank. The updated bank can be evaluated with `--memrl-memory outputs/memrl_online/memory.json`.

## Custom trainer

```bash
python scripts/run_agent.py \
  --agent my_package.agent:Agent \
  --trainer my_package.trainer:Trainer \
  --trainer-state outputs/custom/state \
  --case-ids-file data/splits/train.txt \
  --n 100 --output outputs/custom/train
```

Use a new episode-output directory for each training run. Passing an existing trainer state continues learning from that checkpoint.

## ClinLab

[Train and connect the laboratory predictor](clinlab.md).
