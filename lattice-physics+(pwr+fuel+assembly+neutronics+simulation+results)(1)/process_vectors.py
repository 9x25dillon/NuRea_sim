import pandas as pd
import numpy as np
import math
from pathlib import Path

RAW = Path("raw.csv")
TEST = Path("test.csv")

def load_space_separated_column_csv(path: Path):
    # Load with no header; assume single column with space-separated floats
    df = pd.read_csv(path, header=None)
    rows = df.iloc[:,0].astype(str).tolist()
    parsed = []
    for line in rows:
        # Split on any whitespace, filter empty
        parts = [p for p in line.strip().split() if p]
        try:
            vec = [float(p) for p in parts]
        except Exception:
            # If a row has commas accidentally, try replacing commas with spaces
            parts = [p for p in line.replace(',', ' ').strip().split() if p]
            vec = [float(p) for p in parts]
        parsed.append(vec)
    return parsed

def l2_normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n == 0.0 or not np.isfinite(n):
        return v
    return v / n

def entropy_abs(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    p = np.abs(v)
    s = p.sum()
    if s <= eps:
        return 0.0
    p = p / s
    # Use natural log; units nats
    return float(-np.sum(p * np.log(p + eps)))

def fft_magnitude_feats(v, k=10):
    V = np.fft.fft(np.asarray(v, dtype=np.float64))
    mag = np.abs(V)
    # Use first k components; if shorter, pad with zeros
    if len(mag) < k:
        mag = np.pad(mag, (0, k - len(mag)))
    feats = mag[:k]
    denom = feats.sum()
    if denom == 0 or not np.isfinite(denom):
        return feats
    return feats / denom

def augment_vector(v, k_fft=10):
    v_norm = l2_normalize(v)
    fft_feats = fft_magnitude_feats(v_norm, k=k_fft)
    ent = entropy_abs(v_norm)
    return np.concatenate([v_norm, fft_feats, np.array([ent], dtype=np.float64)])

TARGET_DIM = 1536  # matches VECTOR(1536) in your schema

def pad_or_truncate(vec, target=TARGET_DIM):
    vec = np.asarray(vec, dtype=np.float64)
    if len(vec) >= target:
        return vec[:target]
    return np.pad(vec, (0, target - len(vec)))

def process_file(in_path: Path, out_path: Path, source_name: str):
    vectors = load_space_separated_column_csv(in_path)
    # Determine max base length just for reporting
    base_lengths = [len(v) for v in vectors]
    # Augment and pad
    augmented = [pad_or_truncate(augment_vector(v)) for v in vectors]
    mat = np.vstack(augmented)
    # Create a DataFrame with d1..dTARGET_DIM
    cols = [f"d{i}" for i in range(1, mat.shape[1]+1)]
    df = pd.DataFrame(mat, columns=cols)
    df.to_csv(out_path, index=False)
    return {
        "rows": len(vectors),
        "min_len": int(min(base_lengths) if base_lengths else 0),
        "max_len": int(max(base_lengths) if base_lengths else 0),
        "aug_dim": int(mat.shape[1]),
        "out_file": str(out_path)
    }

raw_out = Path("raw_augmented.csv")
test_out = Path("test_augmented.csv")

print("Processing raw.csv...")
summary_raw = process_file(RAW, raw_out, "raw.csv")
print(f"Raw: {summary_raw}")

print("Processing test.csv...")
summary_test = process_file(TEST, test_out, "test.csv")
print(f"Test: {summary_test}")

print(f"\nOutput files created:")
print(f"  {raw_out} ({summary_raw['rows']} rows, {summary_raw['aug_dim']} dimensions)")
print(f"  {test_out} ({summary_test['rows']} rows, {summary_test['aug_dim']} dimensions)")
print(f"\nReady for Julia ingestion!")
