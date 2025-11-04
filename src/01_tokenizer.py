#!/usr/bin/env python3
# Train SentencePiece tokenizer (Unigram or BPE) at 24k by config.

import argparse, os, sys, yaml
from pathlib import Path
import sentencepiece as spm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["tokenizer"]

    algo = cfg.get("algo", "unigram").lower()  # "unigram" or "bpe"
    assert algo in {"unigram", "bpe"}
    vocab_size = int(cfg.get("vocab_size", 24000))
    character_coverage = float(cfg.get("character_coverage", 0.9995))
    model_prefix = cfg.get("model_prefix", "tokenizer/ka_sp_24000")
    train_input = cfg.get("train_input", "data/splits/train.txt")

    # Optional knobs
    inp_size = int(cfg.get("input_sentence_size", 2_000_000))
    shuffle_inp = bool(cfg.get("shuffle_input_sentence", True))
    num_threads = int(cfg.get("num_threads", 4))

    # Special ids
    unk_id = int(cfg.get("special_ids", {}).get("unk_id", 0))
    bos_id = int(cfg.get("special_ids", {}).get("bos_id", 1))
    eos_id = int(cfg.get("special_ids", {}).get("eos_id", 2))
    pad_id = int(cfg.get("special_ids", {}).get("pad_id", -1))

    Path(os.path.dirname(model_prefix)).mkdir(parents=True, exist_ok=True)

    common = dict(
        input=train_input,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        train_extremely_large_corpus=True,
        input_sentence_size=inp_size,
        shuffle_input_sentence=shuffle_inp,
        num_threads=num_threads,
        unk_id=unk_id,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,  # -1 disables pad
    )

    spm.SentencePieceTrainer.Train(
        model_type=algo,   # "unigram" or "bpe"
        **common
    )

    print(f"[OK] Trained {algo.upper()} tokenizer at {vocab_size}.")
    print(f"     Files: {model_prefix}.model / {model_prefix}.vocab")

if __name__ == "__main__":
    main()
