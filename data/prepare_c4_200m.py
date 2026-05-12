"""
prepare_c4_200m.py
------------------
Downloads a subset of the C4_200M GEC dataset from HuggingFace,
filters bad pairs, splits into train/val/test, and saves as .tsv files.

Usage:
    python prepare_c4_200m.py --size 10000   --out_dir ./data/smoke
    python prepare_c4_200m.py --size 100000  --out_dir ./data/dev
    python prepare_c4_200m.py --size 3000000 --out_dir ./data/final

Output (in out_dir):
    train.tsv   (95%)
    val.tsv     (~3.3%)
    test.tsv    (~1.7%)

Each row in the .tsv: input<TAB>output
"""

import os
import argparse
import random
import time
from datasets import load_dataset
from tqdm import tqdm


# ── Constants ────────────────────────────────────────────────────────────────

SEED           = 42          # Fixed seed for reproducibility across all models
MAX_TOKENS     = 64          # Drop pairs where either side exceeds this (whitespace tokens)
MIN_OVERLAP    = 0.05        # Drop pairs with near-zero token overlap (pure noise)
TRAIN_RATIO    = 0.95
VAL_RATIO      = 0.033
# Test gets the remainder (~0.017)

HF_DATASET     = "liweili/c4_200m"


# ── Helpers ──────────────────────────────────────────────────────────────────

def token_overlap(s1: str, s2: str) -> float:
    """Jaccard overlap between whitespace-tokenized sets of two strings."""
    t1 = set(s1.lower().split())
    t2 = set(s2.lower().split())
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / max(len(t1), len(t2))


def is_valid(inp: str, out: str) -> bool:
    """
    Returns True if the pair passes all filters.

    Filters:
    1. Neither side is empty
    2. Neither side exceeds MAX_TOKENS whitespace tokens
    3. Input and output are not identical (no correction happened)
    4. Token overlap is not near zero (not a pure rewrite / noise)
    """
    if not inp or not out:
        return False
    if len(inp.split()) > MAX_TOKENS or len(out.split()) > MAX_TOKENS:
        return False
    if inp.strip() == out.strip():
        return False
    if token_overlap(inp, out) < MIN_OVERLAP:
        return False
    return True


def split_data(samples: list, seed: int) -> tuple:
    """Shuffle and split into train / val / test."""
    random.seed(seed)
    random.shuffle(samples)

    n = len(samples)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train = samples[:n_train]
    val   = samples[n_train : n_train + n_val]
    test  = samples[n_train + n_val :]

    return train, val, test


def save_tsv(samples: list, path: str):
    """Save list of (input, output) tuples as a TSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for inp, out in samples:
            # Replace any accidental tabs in the text to avoid breaking TSV format
            inp_clean = inp.replace("\t", " ")
            out_clean = out.replace("\t", " ")
            f.write(f"{inp_clean}\t{out_clean}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main(size: int, out_dir: str, hf_token: str = None):

    print(f"\n{'='*55}")
    print(f"  C4_200M Data Preparation")
    print(f"  Target size : {size:,} samples")
    print(f"  Output dir  : {out_dir}")
    print(f"{'='*55}\n")

    # ── 1. Stream from HuggingFace ────────────────────────────────────────
    print("Loading dataset in streaming mode...")
    dataset = load_dataset(
        HF_DATASET,
        split="train",
        streaming=True,
        trust_remote_code=True,
        token=hf_token,
    )

    # ── 2. Stream, filter, collect ────────────────────────────────────────
    print(f"Streaming and filtering up to {size:,} valid samples...\n")

    collected = []
    seen      = 0
    filtered  = 0
    start     = time.time()

    # tqdm bar tracks collected samples toward the target
    with tqdm(total=size, unit="samples", desc="Collecting") as pbar:
        for example in dataset:
            inp = example.get("input", "").strip()
            out = example.get("output", "").strip()
            seen += 1

            if is_valid(inp, out):
                collected.append((inp, out))
                pbar.update(1)
                pbar.set_postfix(streamed=seen, filtered=filtered)
            else:
                filtered += 1

            if len(collected) >= size:
                break

    elapsed = time.time() - start
    mins, secs = divmod(int(elapsed), 60)

    print(f"\nDone streaming in {mins}m {secs}s")
    print(f"  Total streamed : {seen:,}")
    print(f"  Total collected: {len(collected):,}")
    print(f"  Total filtered : {filtered:,} ({100*filtered/seen:.1f}%)")
    print(f"  Throughput     : {seen/elapsed:,.0f} samples/sec\n")

    # ── 3. Split ──────────────────────────────────────────────────────────
    train, val, test = split_data(collected, seed=SEED)

    print(f"Split sizes:")
    print(f"  Train : {len(train):,} ({100*len(train)/len(collected):.1f}%)")
    print(f"  Val   : {len(val):,}  ({100*len(val)/len(collected):.1f}%)")
    print(f"  Test  : {len(test):,}  ({100*len(test)/len(collected):.1f}%)\n")

    # ── 4. Save ───────────────────────────────────────────────────────────
    train_path = os.path.join(out_dir, "train.tsv")
    val_path   = os.path.join(out_dir, "val.tsv")
    test_path  = os.path.join(out_dir, "test.tsv")

    save_tsv(train, train_path)
    save_tsv(val,   val_path)
    save_tsv(test,  test_path)

    print(f"Saved:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")

    # ── 5. Quick sanity check ─────────────────────────────────────────────
    print(f"\nSanity check — first 3 rows of train.tsv:")
    for inp, out in train[:3]:
        print(f"  IN : {inp[:80]}")
        print(f"  OUT: {out[:80]}")
        print()

    print("✅ Preparation complete.\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare C4_200M GEC subsets")
    parser.add_argument("--size",    type=int, required=True,  help="Number of valid samples to collect")
    parser.add_argument("--out_dir", type=str, required=True,  help="Output directory for .tsv files")
    parser.add_argument("--hf_token",type=str, default=None,   help="HuggingFace token (optional if already set in env)")
    args = parser.parse_args()

    # Try env variable if token not passed directly
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", None)

    main(size=args.size, out_dir=args.out_dir, hf_token=hf_token)