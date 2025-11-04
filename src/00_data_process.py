#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean and split Georgian corpora from data/raw to data/cleaned and data/splits.

This script:
  1) Loads config (YAML) for paths and cleaning/splitting parameters.
  2) Streams all *.txt files under raw_dir, applies cleaning rules.
  3) Performs exact dedup (and optional light near-dup filtering).
  4) Optionally appends <eos> to each kept line.
  5) Writes a merged cleaned corpus and train/val/test splits.
  6) Saves a JSON stats report (counts per file and reasons of removal).

Notes:
- "Georgian ratio" is computed as (#Georgian letters) / (#all letters).
- Unicode ranges considered Georgian: U+10A0–U+10FF (Georgian),
  U+1C90–U+1CBF (Georgian Extended).
- Keep the near-duplicate option off by default; it is simple and may be slow on huge corpora.
"""

import argparse
import json
import os
import re
import sys
import unicodedata
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

try:
    import yaml
except Exception as e:
    print("Please pip install pyyaml first.", file=sys.stderr)
    raise

try:
    from tqdm import tqdm
except Exception:
    # Fallback if tqdm is not available
    def tqdm(x, **kwargs):
        return x

# --- Regexes and helpers for cleaning ---
RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
RE_HTML = re.compile(r"<[^>]+>")
# Georgian letters: Mkhedruli + Asomtavruli + Mtavruli blocks
RE_GEORGIAN = re.compile(r"[\u10A0-\u10FF\u1C90-\u1CBF]")
RE_DIGITS = re.compile(r"\d")

# NEW: line-number prefix at start of line:
#   matches: "6 ", "6) ", "6:", "6 ! ", '6 !" ', "15 !01 ..." (keeps "01")
RE_LEADING_LINENO = re.compile(
    r"^\s*\d+(?:[.)]|:)?\s*(?:[!\"“”„\-–—])?\s*"
)

def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def ensure_dirs(*paths: str) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def iter_raw_files(raw_dir: str) -> List[Path]:
    p = Path(raw_dir)
    files = sorted(p.glob("*.txt"))
    return files

def normalize_text(s: str, form: str = "NFKC", to_lower: bool = False) -> str:
    s = unicodedata.normalize(form, s)
    if to_lower:
        s = s.lower()
    return s

def strip_noise(s: str, strip_html: bool = True, strip_urls: bool = True) -> str:
    if strip_html:
        s = RE_HTML.sub(" ", s)
    if strip_urls:
        s = RE_URL.sub(" ", s)
    # Collapse spaces
    s = " ".join(s.split())
    return s

def strip_leading_lineno(s: str) -> Tuple[str, bool]:
    """
    Remove a leading line number and optional punctuation from the start of the line.
    Returns (new_string, stripped:bool).
    Examples handled:
      "6 01 ..." -> "01 ..."
      "7 ! 01 ..." -> "01 ..."
      "15 !01 ..." -> "01 ..."
      "12) «... »" -> "«... »"
      "3: text" -> "text"
    """
    new_s, n = RE_LEADING_LINENO.subn("", s, count=1)
    return new_s, (n > 0)

def georgian_ratio(s: str) -> float:
    """Compute ratio of Georgian letters over all alphabetic letters."""
    letters_total = sum(1 for ch in s if ch.isalpha())
    if letters_total == 0:
        return 0.0
    geos = len(RE_GEORGIAN.findall(s))
    return geos / max(1, letters_total)

def digits_ratio(s: str) -> float:
    """Compute ratio of digits over total characters (excluding spaces)."""
    no_space = s.replace(" ", "")
    if not no_space:
        return 0.0
    return len(RE_DIGITS.findall(no_space)) / len(no_space)

def exact_hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

# --- Optional light near-duplicate check (token 5-gram Jaccard) ---
def token_ngrams(s: str, n: int = 5) -> set:
    toks = s.split()
    if len(toks) < n:
        return set()
    return {" ".join(toks[i:i+n]) for i in range(len(toks)-n+1)}

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / max(1, union)

def should_keep_line(
    s: str,
    min_chars: int,
    max_digits_ratio: float,
    min_geo_ratio: float
) -> Tuple[bool, str]:
    """Return (keep, reason_if_dropped)."""
    if len(s) < min_chars:
        return False, "too_short"
    if digits_ratio(s) > max_digits_ratio:
        return False, "too_numeric"
    if georgian_ratio(s) < min_geo_ratio:
        return False, "low_geo_ratio"
    return True, ""

def clean_stream(
    files: List[Path],
    cfg_clean: Dict
) -> Tuple[List[str], Dict]:
    """
    Clean and deduplicate all lines from files.
    Returns (kept_lines, stats_dict).
    """
    normalize_form = cfg_clean.get("normalize", "NFKC")
    to_lower = bool(cfg_clean.get("lowercase", False))
    strip_html = bool(cfg_clean.get("strip_html", True))
    strip_urls = bool(cfg_clean.get("strip_urls", True))
    min_chars = int(cfg_clean.get("min_chars", 15))
    max_digits_ratio = float(cfg_clean.get("max_digits_ratio", 0.40))
    min_geo_ratio = float(cfg_clean.get("min_georgian_ratio", 0.85))
    dedupe = bool(cfg_clean.get("deduplicate", True))

    # NEW: toggle for stripping leading line numbers
    strip_lineno = bool(cfg_clean.get("strip_leading_lineno", True))

    near = cfg_clean.get("near_dedupe", {}) or {}
    near_enabled = bool(near.get("enabled", False))
    near_j = float(near.get("jaccard_threshold", 0.90))
    near_n = int(near.get("ngram", 5))
    near_prefix = int(near.get("bucket_prefix_len", 16))

    stats = {
        "per_file": {},
        "total_raw_lines": 0,
        "kept_lines": 0,
        "dropped": {
            "too_short": 0,
            "too_numeric": 0,
            "low_geo_ratio": 0,
            "empty_after_clean": 0,
        },
        "dedup_removed": 0,
        "near_dedup_removed": 0,
        "stripped_leading_lineno": 0,  # NEW
        "config": {
            "min_chars": min_chars,
            "max_digits_ratio": max_digits_ratio,
            "min_georgian_ratio": min_geo_ratio,
            "deduplicate": dedupe,
            "strip_leading_lineno": strip_lineno,  # NEW
            "near_dedupe": {
                "enabled": near_enabled,
                "jaccard_threshold": near_j,
                "ngram": near_n,
                "bucket_prefix_len": near_prefix,
            }
        }
    }

    seen_hashes = set()
    # Buckets for light near-dup (prefix-based)
    near_buckets: Dict[str, List[Tuple[str, set]]] = {}
    kept: List[str] = []

    for fp in files:
        pf = str(fp)
        pf_stats = {"raw_lines": 0, "kept": 0}
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                stats["total_raw_lines"] += 1
                pf_stats["raw_lines"] += 1
                s = line.rstrip("\n\r")
                if not s.strip():
                    stats["dropped"]["empty_after_clean"] += 1
                    continue

                # --- NEW: strip leading line number BEFORE normalization/noise ---
                if strip_lineno:
                    s2, stripped = strip_leading_lineno(s)
                    if stripped:
                        stats["stripped_leading_lineno"] += 1
                    s = s2

                # Normalize, noise stripping, and optional lower-casing
                s = normalize_text(s, form=normalize_form, to_lower=to_lower)
                s = strip_noise(s, strip_html=strip_html, strip_urls=strip_urls)
                if not s:
                    stats["dropped"]["empty_after_clean"] += 1
                    continue

                keep, reason = should_keep_line(
                    s,
                    min_chars=min_chars,
                    max_digits_ratio=max_digits_ratio,
                    min_geo_ratio=min_geo_ratio
                )
                if not keep:
                    stats["dropped"][reason] += 1
                    continue

                if dedupe:
                    h = exact_hash(s)
                    if h in seen_hashes:
                        stats["dedup_removed"] += 1
                        continue
                    seen_hashes.add(h)

                if near_enabled:
                    # Bucket by a normalized prefix to avoid O(N^2)
                    prefix = s[:near_prefix]
                    cand_list = near_buckets.setdefault(prefix, [])
                    # Compare only within this small bucket
                    cand_ng = token_ngrams(s, n=near_n)
                    is_dup = False
                    for cand_s, cand_set in cand_list:
                        if cand_set and cand_ng:
                            if jaccard(cand_set, cand_ng) >= near_j:
                                stats["near_dedup_removed"] += 1
                                is_dup = True
                                break
                    if is_dup:
                        continue
                    cand_list.append((s, cand_ng))

                kept.append(s)
                pf_stats["kept"] += 1

        stats["per_file"][pf] = pf_stats

    stats["kept_lines"] = len(kept)
    return kept, stats

def append_eos(lines: List[str], eos_token: str) -> List[str]:
    out = []
    tail = " " + eos_token if eos_token else ""
    for s in lines:
        out.append(s + tail)
    return out

def write_text(path: str, lines: Iterable[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in lines:
            f.write(s)
            f.write("\n")

def do_split(
    lines: List[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    shuffle: bool,
    out_dir: str
) -> Dict:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    n = len(lines)
    idx = list(range(n))
    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]

    train = [lines[i] for i in train_idx]
    val = [lines[i] for i in val_idx]
    test = [lines[i] for i in test_idx]

    ensure_dirs(out_dir)
    write_text(str(Path(out_dir) / "train.txt"), train)
    write_text(str(Path(out_dir) / "val.txt"), val)
    write_text(str(Path(out_dir) / "test.txt"), test)

    return {
        "train": len(train),
        "val": len(val),
        "test": len(test),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw_dir = cfg["paths"]["raw_dir"]
    cleaned_dir = cfg["paths"]["cleaned_dir"]
    splits_dir = cfg["paths"]["splits_dir"]

    outputs = cfg.get("outputs", {})
    merged_out = outputs.get("cleaned_merged", str(Path(cleaned_dir) / "ka_cleaned.txt"))
    stats_path = outputs.get("stats_path", str(Path(cleaned_dir) / "clean_stats.json"))

    split_cfg = cfg.get("split", {})
    train_ratio = float(split_cfg.get("train_ratio", 0.98))
    val_ratio = float(split_cfg.get("val_ratio", 0.01))
    test_ratio = float(split_cfg.get("test_ratio", 0.01))
    seed = int(split_cfg.get("seed", 1337))
    shuffle = bool(split_cfg.get("shuffle_before_split", True))

    clean_cfg = cfg.get("cleaning", {})

    ensure_dirs(cleaned_dir, splits_dir)
    files = iter_raw_files(raw_dir)
    if not files:
        print(f"No .txt files found under {raw_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Found {len(files)} raw files:")
    for fp in files:
        print(" -", fp.name)

    print("[INFO] Cleaning & deduplicating ...")
    kept, stats = clean_stream(files, clean_cfg)

    if clean_cfg.get("append_eos", True):
        eos = clean_cfg.get("eos_token", "<eos>")
        kept = append_eos(kept, eos)

    print(f"[INFO] Writing merged cleaned corpus to {merged_out}")
    write_text(merged_out, kept)

    print("[INFO] Splitting into train/val/test ...")
    split_counts = do_split(
        kept, train_ratio, val_ratio, test_ratio, seed, shuffle, splits_dir
    )
    stats["split_counts"] = split_counts

    print(f"[INFO] Writing stats to {stats_path}")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("[DONE]")
    print(json.dumps({
        "total_raw_lines": stats["total_raw_lines"],
        "kept_lines": stats["kept_lines"],
        "dropped": stats["dropped"],
        "dedup_removed": stats["dedup_removed"],
        "near_dedup_removed": stats["near_dedup_removed"],
        "stripped_leading_lineno": stats["stripped_leading_lineno"],
        "split_counts": stats["split_counts"]
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
