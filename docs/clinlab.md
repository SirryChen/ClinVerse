# ClinLab training

## Environment and base model

Use the ClinVerse Conda environment on a CUDA machine. The reference LoRA configuration uses one 48 GB GPU, bfloat16, sequence length 4,096, batch size 1 and gradient checkpointing.

```bash
pip install -r requirements-gpu.txt
hf download Qwen/Qwen3.5-2B --local-dir weights/Qwen3.5-2B
```

Base model: [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B).

## Build training records

After data normalization:

```bash
python training/provider/mimic_iv_ext/prepare_data.py --config config/clinlab.yaml
```

This writes admission-level train/test records under `outputs/clinlab/data/`.

## Train

```bash
python training/provider/mimic_iv_ext/train.py --config config/clinlab.yaml
```

The final adapter is written to `outputs/clinlab/checkpoints/final_adapter/`. Training parameters and source-table paths are in `config/clinlab.yaml`.

## Connect to the environment

```bash
python scripts/configure_local_lab.py \
  --model weights/Qwen3.5-2B \
  --adapter outputs/clinlab/checkpoints/final_adapter \
  --mapping data/normalized/mimic-iv-ext/lab_test_mapping.csv \
  --output config/local.yaml

python scripts/run_agent.py \
  --config config/local.yaml \
  --agent examples.openai_agent:OpenAIAgent \
  --case-ids-file data/splits/test.txt \
  --n 100 \
  --output outputs/evaluation/local_clinlab
```

The patient and doctor use their configured endpoints. ClinLab loads the local base model and adapter on `cuda:0`.
