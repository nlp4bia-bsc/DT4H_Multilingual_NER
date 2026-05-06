# DT4H NER Models

This repository contains code and documentation for multilingual Named Entity Recognition (NER) models developed within the DT4H (Data Tools for Heart) Europe Horizon project.

The focus is multilingual clinical NER for disease, symptom, and procedure mention detection.

## Repository structure

- `training_notebooks/`: training notebooks used during model development.
  - **Multilingual multilabel MTL** (one head per entity type across languages in one stream): `training_notebooks/MultiClinAI_MTL_Multilingual_Multilabel_NER_Training_Script.ipynb`
  - **Multilingual monolabel MTL** (one head per language, same entity type, languages belonging to the same linguistic family): `training_notebooks/MultiClinAI_MTL_Multilingual_Monolabel_NER_Training_Script.ipynb` (same pattern works for different language triples, e.g. `en–nl–sv`, `es-it-ro`).
  - **Multilingual monolabel STL** (shared encoder trained on labeled data across all languages for a given entity type) `training_notebooks/MultiClinAI_STL_Multilingual_Monolabel_NER_Training_Script.ipynb`
- `inference_scripts/`: two simple Python scripts for terminal inference. Both scripts generate a single `.tsv` file containing predictions formatted for MultiClinNER evaluation. The output `.tsv` file is saved in the same directory as the input data directory.
  - `predict_mentions_stl_hf.py`: inference for Hugging Face single-task models.
  - `predict_mentions_mtl_local.py`: inference for local `.pt` multi-task models.
- `requirements.txt`: dependencies for running inference scripts.

## Models covered

### Single-task (Hugging Face)

- [`BSC-NLP4BIA/DT4H_XLM-R_stl_multilingual_disease`](https://huggingface.co/BSC-NLP4BIA/DT4H_XLM-R_stl_multilingual_disease)
- [`BSC-NLP4BIA/DT4H_XLM-R_stl_multilingual_symptom`](https://huggingface.co/BSC-NLP4BIA/DT4H_XLM-R_stl_multilingual_symptom)
- [`BSC-NLP4BIA/DT4H_XLM-R_stl_multilingual_procedure`](https://huggingface.co/BSC-NLP4BIA/DT4H_XLM-R_stl_multilingual_procedure)

### Multi-task (local .pt)

Group A (`es-it-ro`):
- [`BSC-NLP4BIA/DT4H_XLM-R_mtl_es-it-ro_disease.pt`](https://huggingface.co/BSC-NLP4BIA/DT4H_XLM-R_mtl_es-it-ro_disease)
- [`BSC-NLP4BIA/DT4H_XLM-R_mtl_es-it-ro_symptom.pt`](https://huggingface.co/BSC-NLP4BIA/DT4H_XLM-R_mtl_es-it-ro_symptom)
- [`BSC-NLP4BIA/DT4H_XLM-R_mtl_es-it-ro_procedure.pt`](https://huggingface.co/BSC-NLP4BIA/DT4H_XLM-R_mtl_es-it-ro_procedure)

Group B (`en-nl-sv`):
- [`BSC-NLP4BIA/DT4H_XLM-R_mtl_en-nl-sv_disease.pt`](https://huggingface.co/BSC-NLP4BIA/DT4H_XLM-R_mtl_en-nl-sv_disease)
- [`BSC-NLP4BIA/DT4H_XLM-R_mtl_en-nl-sv_symptom.pt`](https://huggingface.co/BSC-NLP4BIA/DT4H_XLM-R_mtl_en-nl-sv_symptom)
- [`BSC-NLP4BIA/DT4H_XLM-R_mtl_en-nl-sv_procedure.pt`](https://huggingface.co/BSC-NLP4BIA/DT4H_XLM-R_mtl_en-nl-sv_procedure)

Multilingual multilabel:
- [`BSC-NLP4BIA/DT4H_XLM-R_mtl_multilingual_multilabel.pt`](https://huggingface.co/BSC-NLP4BIA/DT4H_XLM-R_mtl_multilingual_multilabel)

All models are publicly available in BSC-NLP4BIA’s “DT4H Multilingual NER Models” Hugging Face collection: https://huggingface.co/collections/BSC-NLP4BIA/dt4h-multilingual-ner-models

## Quick usage

```bash
pip install -r requirements.txt
```

### 1) Single-task model inference

```bash
python inference_scripts/predict_mentions_stl_hf.py \
  --input_dir /path/to/txt_files \
  --model_name BSC-NLP4BIA/DT4H_XLM-R_stl_multilingual_disease \
  --output_tsv outputs_disease_es.tsv
```

#### Arguments

| Argument | Description |
|---|---|
| `--input_dir` | Directory containing the input `.txt` files for inference. |
| `--model_name` | Hugging Face model repository name to use for inference. |
| `--output_tsv` | Name or path of the output `.tsv` prediction file. |

### 2) Multi-task local model inference

```bash
python inference_scripts/predict_mentions_mtl_local.py \
  --input_dir /path/to/txt_files \
  --model_path /path/to/DT4H_XLM-R_mtl_es-it-ro_disease.pt \
  --task disease_ro_ner \
  --output_tsv outputs_disease_ro.tsv
```

#### Arguments

| Argument | Description |
|---|---|
| `--input_dir` | Directory containing the input `.txt` files for inference. |
| `--model_path` | Path to the local multi-task .pt model checkpoint. |
| `--task` | Which prediction head to run (must match the training configuration). Available tasks for each model are listed in the corresponding **Hugging Face model card**. Example: `disease`. |
| `--output_tsv` | Name or path of the output `.tsv` prediction file. |

### Output format:

```text
filename    label    start_span    end_span    text
```

All predictions are exported into a **single TSV file**.
