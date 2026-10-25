# Data processing

## Sources

- [MIMIC-IV 3.1](https://physionet.org/content/mimiciv/3.1/)
- [MIMIC-IV-Note 2.2](https://physionet.org/content/mimic-iv-note/2.2/)
- [SICdb 1.0.8](https://physionet.org/content/sicdb/1.0.8/)
- [Synthea](https://github.com/synthetichealth/synthea)

The MIMIC command expects `hosp/*.csv.gz` under the MIMIC directory and `note/*.csv.gz` under the Note directory. Download these files from the corresponding source pages.

## Setup

```bash
conda env create -f environment.yml
conda activate clinverse
```

Configure the `extractor` endpoint in `.env`. Other endpoints are used during evaluation and training. The information extractor produces structured, patient-reportable facts with category, source, importance and visibility fields.

## End-to-end processing

```bash
python scripts/prepare_data.py \
  --mimic-dir ../dataset/mimic-iv-3.1 \
  --note-dir ../dataset/mimic-iv-note/2.2 \
  --synthea-root ../synthea \
  --setup-synthea \
  --extractor-provider extractor \
  --resume
```

| Stage | Output |
|---|---|
| Normalize admissions, notes and measurements | `data/normalized/mimic-iv-ext/` |
| Build examination catalogs | `lab_provider_catalog.json`, `radiology_catalog.json` |
| Generate backgrounds and information units | `data/cases/*.json`, `index.jsonl` |
| Record configuration and progress | `data/preparation/` |

Use `--limit 100` to restrict the admission set, `--copies 10` to set the candidate count, and `--batch-size 20` to control each generation batch. `--resume` continues the same processing run. The default Synthea revision is `29a7d63ce80e`. `--synthea-ref` selects another revision for a new checkout; the resolved revision is recorded in `data/preparation/synthea.json`.

Inspect the command sequence:

```bash
python scripts/prepare_data.py \
  --mimic-dir ../dataset/mimic-iv-3.1 \
  --note-dir ../dataset/mimic-iv-note/2.2 \
  --plan
```

## Splits and validation

```bash
python scripts/split_cases.py \
  --cases-dir data/cases \
  --output data/splits \
  --seed 2026

python scripts/audit_cases.py --cases-dir data/cases
```

The split command groups admissions by source and patient, then writes `train.txt`, `validation.txt`, `test.txt` and a split manifest. The default patient proportions are 80%, 10% and 10%.

## Existing Synthea checkout

```bash
python scripts/setup_synthea.py --synthea-root ../synthea
```

Then run `prepare_data.py` without `--setup-synthea`.

## SICdb

The SICdb preprocessing code is in `preprocessing/sicdb_ext/predata/`. Its normalized output uses the same dataset-adapter and case interfaces. Configure its source directory, then run:

```bash
export CLINVERSE_SICDB_DIR=../dataset/SICdb
export CLINVERSE_SICDB_NORMALIZED_DIR=data/normalized/sicdb-ext

python preprocessing/sicdb_ext/predata/run.py
python preprocessing/sicdb_ext/predata/build_catalog.py
python scripts/build_conditioned_cases.py \
  --dataset sicdb-ext \
  --info-extractor-provider extractor \
  --synthea-root ../synthea \
  --output data/cases_sicdb
```
