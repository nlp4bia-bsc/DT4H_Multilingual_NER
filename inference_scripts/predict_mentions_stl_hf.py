import argparse
import os
import unicodedata
from collections import Counter, defaultdict

import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


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


def normalize_id2label(model):
    """
    Ensure HF id2label keys are integers.
    Some checkpoints store keys as strings in config.json.
    """
    id2label_model = model.config.id2label
    if id2label_model:
        return {int(k): v for k, v in id2label_model.items()}

    # Fallback if id2label is missing: build from label2id
    label2id_model = getattr(model.config, "label2id", None) or {}
    return {int(v): k for k, v in label2id_model.items()}


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
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_ids = torch.argmax(outputs.logits, dim=-1).squeeze(0).tolist()

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
            raw_label = id2label[task].get(pred_ids[first_i], "O")
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


def main():
    parser = argparse.ArgumentParser(description="Run STL HF model on a folder of txt files.")
    parser.add_argument("--input_dir", required=True, help="Path to directory with .txt files")
    parser.add_argument("--model_name", required=True, help="Hugging Face model name")
    parser.add_argument("--output_tsv", required=True, help="Output TSV path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name).to(device)
    model.eval()

    id2label_model = normalize_id2label(model)
    id2label = {"hf_model_task": id2label_model}
    task = "hf_model_task"

    rows = []
    for txt_file in get_txt_files(args.input_dir):
        with open(txt_file, "r", encoding="utf-8") as f:
            txt = f.read()
        entities = predict_entities(txt, task, model, tokenizer, device, id2label)
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
    df.to_csv(args.output_tsv, sep="\t", index=False)
    print(f"Saved: {args.output_tsv}")


if __name__ == "__main__":
    main()
