"""
Multi-task NER inference for local .pt checkpoints from:

  A) MultiClinAI_MTL_Multilingual_Multilabel_NER_Training_Script — heads keyed
     disease / symptom / procedure (nn.ModuleDict in state_dict: classifiers.<task>).

  B) MultiClinAI_MTL_NER_es_ro_it_disease (and same pattern for en-nl-sv, etc.) —
     heads as disease_es_classifier, … in state_dict; training uses task strings
     disease_es_ner, disease_it_ner, disease_ro_ner (or disease_en_ner, …).

Usage: pass --input_dir, --model_path, --task, and optionally --model_checkpoint.
Use --list_tasks to print head names found in the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


def is_edge_punct(ch):
    return unicodedata.category(ch).startswith("P")


def trim_span_punctuation(txt, start, end):
    while start < end and txt[start].isspace():
        start += 1
    while start < end and txt[end - 1].isspace():
        end -= 1
    while start < end and is_edge_punct(txt[start]):
        start += 1
    while start < end and is_edge_punct(txt[end - 1]):
        end -= 1
    return start, end


def flush_entity(entities, txt, current_entity, start_pos, end_pos):
    if current_entity is not None and start_pos is not None and end_pos is not None:
        start_pos, end_pos = trim_span_punctuation(txt, start_pos, end_pos)
        if start_pos < end_pos:
            entities.append({
                "label": current_entity,
                "start": start_pos,
                "end": end_pos,
                "text": txt[start_pos:end_pos],
            })


def predict_entities(txt, task, model, tokenizer, device, id2label, max_length=512, stride=128):
    enc = tokenizer(
        txt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        stride=stride,
        padding="max_length",
    )

    span_votes = defaultdict(list)
    n_windows = enc["input_ids"].shape[0]

    for w in range(n_windows):
        input_ids = enc["input_ids"][w].unsqueeze(0).to(device)
        attention_mask = enc["attention_mask"][w].unsqueeze(0).to(device)
        offset_mapping = enc["offset_mapping"][w].tolist()
        word_ids = enc.word_ids(batch_index=w)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, task=task)
            pred_ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()

        word_to_indices = defaultdict(list)
        for i, wid in enumerate(word_ids):
            if wid is None:
                continue
            word_to_indices[wid].append(i)

        for wid in sorted(word_to_indices.keys()):
            idxs = word_to_indices[wid]
            first_i, last_i = idxs[0], idxs[-1]
            s_ch, e_ch = offset_mapping[first_i][0], offset_mapping[last_i][1]
            if s_ch == 0 and e_ch == 0:
                continue
            raw_label = id2label.get(pred_ids[first_i], "O")
            span_votes[(s_ch, e_ch)].append(raw_label)

    final_words = []
    for (s_ch, e_ch), labels in span_votes.items():
        counts = Counter(labels).most_common()
        top_count = counts[0][1]
        top_labels = [lab for lab, cnt in counts if cnt == top_count]
        if len(top_labels) > 1 and "O" in top_labels:
            top_labels.remove("O")
        final_words.append((s_ch, e_ch, top_labels[0]))

    final_words.sort(key=lambda x: (x[0], x[1]))

    entities = []
    current_entity, start_pos, end_pos = None, None, None

    for s_ch, e_ch, raw in final_words:
        if raw.startswith("B-"):
            etype = raw[2:]
            flush_entity(entities, txt, current_entity, start_pos, end_pos)
            current_entity, start_pos, end_pos = etype, s_ch, e_ch
        elif raw.startswith("I-"):
            etype = raw[2:]
            if current_entity == etype:
                end_pos = e_ch
        else:
            flush_entity(entities, txt, current_entity, start_pos, end_pos)
            current_entity, start_pos, end_pos = None, None, None

    flush_entity(entities, txt, current_entity, start_pos, end_pos)
    return entities


def get_txt_files(base_dir):
    txt_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".txt"):
                txt_files.append(os.path.join(root, f))
    return sorted(txt_files)


ENTITY_TYPES = ("disease", "symptom", "procedure")


def default_id2label_three_bio(entity: str) -> Dict[int, str]:
    u = entity.strip().upper()
    return {0: f"B-{u}", 1: f"I-{u}", 2: "O"}


def infer_entity_from_task(task: str) -> str:
    if task in ENTITY_TYPES:
        return task
    m = re.match(r"^([a-z]+)_[a-z]{2}_ner$", task)
    if m:
        return m.group(1)
    # Legacy ModuleDict keys without _ner (e.g. older local training / inference scripts)
    m = re.match(r"^([a-z]+)_[a-z]{2}$", task)
    if m:
        return m.group(1)
    if task.endswith("_ner"):
        body = task[: -len("_ner")]
        if "_" in body:
            return body.rsplit("_", 1)[0]
    raise ValueError(
        f"Cannot infer entity type from task {task!r}. "
        f"Use one of {list(ENTITY_TYPES)} or *_<lang>_ner, or pass --id2label_json."
    )


def build_id2label_for_task(task: str, num_labels: int, json_path: Optional[str]) -> Dict[int, str]:
    if json_path:
        with open(json_path, encoding="utf-8") as f:
            raw = json.load(f)
        out = {int(k): v for k, v in raw.items()}
        if len(out) != num_labels:
            raise ValueError(f"id2label JSON has {len(out)} labels, head expects {num_labels}")
        return out
    if num_labels != 3:
        raise ValueError(
            f"Head {task!r} has num_labels={num_labels}; provide --id2label_json with keys 0..{num_labels - 1}."
        )
    entity = infer_entity_from_task(task)
    if entity not in ENTITY_TYPES:
        raise ValueError(
            f"Inferred entity {entity!r} from task {task!r} is not in {ENTITY_TYPES}. "
            "Pass --id2label_json if labels differ."
        )
    return default_id2label_three_bio(entity)


def strip_optional_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keys = list(state_dict.keys())
    if keys and all(k.startswith("model.") for k in keys):
        return {k[len("model.") :]: v for k, v in state_dict.items()}
    return dict(state_dict)


def discover_heads_from_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, int]:
    heads: Dict[str, int] = {}

    for k, tensor in sd.items():
        if not hasattr(tensor, "shape") or len(tensor.shape) != 2:
            continue
        if k.startswith("encoder.") or k.startswith("roberta.") or k.startswith("bert."):
            continue

        if k.startswith("classifiers.") and k.endswith(".weight"):
            parts = k.split(".")
            if len(parts) == 3 and parts[0] == "classifiers":
                task = parts[1]
                heads[task] = int(tensor.shape[0])
            continue

        m = re.match(r"^(.+)_classifier\.weight$", k)
        if m:
            prefix = m.group(1)
            task = f"{prefix}_ner"
            heads[task] = int(tensor.shape[0])

    return heads


def remap_state_dict_for_module_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("encoder.") or k.startswith("roberta.") or k.startswith("bert."):
            out[k] = v
            continue
        if k.startswith("classifiers."):
            out[k] = v
            continue
        m = re.match(r"^(.+)_classifier\.(weight|bias)$", k)
        if m:
            prefix, wb = m.groups()
            out[f"classifiers.{prefix}_ner.{wb}"] = v
        else:
            out[k] = v
    return out


class MultiTaskNERInferenceModel(nn.Module):
    def __init__(self, model_checkpoint: str, head_num_labels: Dict[str, int]):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_checkpoint)
        self.dropout = nn.Dropout(0.1)
        hidden = self.encoder.config.hidden_size
        self.classifiers = nn.ModuleDict({
            task: nn.Linear(hidden, n_labels) for task, n_labels in head_num_labels.items()
        })

    def forward(self, input_ids, attention_mask, task: str):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        if task not in self.classifiers:
            avail = ", ".join(sorted(self.classifiers.keys()))
            raise ValueError(f"Unknown task {task!r}. Available heads: {avail}")
        return self.classifiers[task](sequence_output)


def load_model_for_inference(
    model_path: str,
    model_checkpoint: str,
) -> Tuple[MultiTaskNERInferenceModel, List[str]]:
    raw = torch.load(model_path, map_location="cpu")
    if isinstance(raw, nn.Module):
        raise ValueError(
            "Checkpoint is a full nn.Module pickle. This script expects state_dict "
            "(e.g. torch.save(model.state_dict(), path))."
        )
    sd = strip_optional_prefix(raw)
    heads = discover_heads_from_state_dict(sd)
    if not heads:
        raise ValueError(
            "Could not find classifier heads in checkpoint. Expected keys like "
            "'classifiers.<task>.weight' or 'disease_es_classifier.weight'."
        )
    sd_norm = remap_state_dict_for_module_dict(sd)
    model = MultiTaskNERInferenceModel(model_checkpoint, heads)
    missing, unexpected = model.load_state_dict(sd_norm, strict=False)
    if missing:
        clf_missing = [m for m in missing if m.startswith("classifiers.")]
        if clf_missing:
            raise RuntimeError(f"Missing classifier weights after load: {clf_missing}")
        enc_missing = [m for m in missing if m.startswith("encoder.")]
        if enc_missing:
            raise RuntimeError(
                f"Missing encoder weights ({len(enc_missing)} keys). "
                "Use --model_checkpoint matching the training backbone."
            )
    if unexpected:
        rest = [u for u in unexpected if not (u.startswith("encoder.") and "pooler" in u)]
        if rest:
            tail = "..." if len(rest) > 20 else ""
            print(f"Warning: unexpected keys in checkpoint (ignored): {rest[:20]}{tail}")
    return model, sorted(heads.keys())


def main():
    parser = argparse.ArgumentParser(
        description="MTL NER inference: one .pt checkpoint, one --task head, folder of .txt files."
    )
    parser.add_argument(
        "--input_dir",
        help="Directory tree containing .txt files (omit with --list_tasks)",
    )
    parser.add_argument("--model_path", required=True, help="Local .pt file (state_dict)")
    parser.add_argument(
        "--task",
        help=(
            "Head name: multilingual-multilabel → disease | symptom | procedure. "
            "Per-language MTL → disease_es_ner, disease_it_ner, disease_en_ner, …"
        ),
    )
    parser.add_argument("--output_tsv", help="Output TSV path (omit with --list_tasks)")
    parser.add_argument("--model_checkpoint", default="FacebookAI/xlm-roberta-base")
    parser.add_argument(
        "--id2label_json",
        default=None,
        help='Optional JSON {"0": "B-DISEASE", "1": "I-DISEASE", "2": "O"} if label ids differ.',
    )
    parser.add_argument("--list_tasks", action="store_true", help="Print head names in checkpoint and exit")
    args = parser.parse_args()

    model, tasks = load_model_for_inference(args.model_path, args.model_checkpoint)
    if args.list_tasks:
        print("Heads in checkpoint:", ", ".join(tasks))
        return
    if not args.input_dir or not args.output_tsv or not args.task:
        parser.error("--input_dir, --output_tsv, and --task are required unless you pass --list_tasks.")
    if args.task not in tasks:
        raise SystemExit(
            f"Task {args.task!r} not in checkpoint. Heads: {', '.join(tasks)}. "
            "Re-run with --list_tasks."
        )

    num_labels = model.classifiers[args.task].out_features
    id2label = build_id2label_for_task(args.task, num_labels, args.id2label_json)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=True)

    rows = []
    for txt_file in get_txt_files(args.input_dir):
        with open(txt_file, "r", encoding="utf-8") as f:
            txt = f.read()
        entities = predict_entities(txt, args.task, model, tokenizer, device, id2label)
        fname = os.path.splitext(os.path.basename(txt_file))[0]
        for ent in entities:
            rows.append({
                "filename": fname,
                "label": ent["label"],
                "start_span": ent["start"],
                "end_span": ent["end"],
                "text": ent["text"],
            })

    df = pd.DataFrame(rows, columns=["filename", "label", "start_span", "end_span", "text"])
    df.to_csv(args.output_tsv, sep="	", index=False)
    print(f"Saved: {args.output_tsv}")


if __name__ == "__main__":
    main()
