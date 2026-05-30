"""
Generate a Zipf-skewed query stream for the bench.

Input:  SIFT query file (binary: <n:u32><d:u32><n*d float32>)
Output: a stream where 1M total queries are sampled from a small "hot" pool
        (100 distinct) with Zipf(α) frequencies, plus a "cold" tail.

This simulates production traffic where popular queries dominate.
"""
import argparse, struct, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp",
                default="../data/sift/sift_query_stream.bin")
ap.add_argument("--out", default="../data/sift/sift_query_zipf.bin")
ap.add_argument("--hot_pool", type=int, default=100)
ap.add_argument("--total", type=int, default=1_000_000)
ap.add_argument("--alpha", type=float, default=1.5)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

# read header + vectors
with open(args.inp, "rb") as f:
    n, d = struct.unpack("II", f.read(8))
    vecs = np.frombuffer(f.read(n*d*4), dtype=np.float32).reshape(n, d)
print(f"loaded {n} queries dim {d}")

# Sample hot_pool from first N queries (deterministic)
rng = np.random.default_rng(args.seed)
hot = vecs[:args.hot_pool]

# Generate Zipf samples (in [1, ...]); clip to [0, hot_pool)
samples = rng.zipf(args.alpha, size=args.total * 3)
samples = samples[samples <= args.hot_pool][:args.total]
if samples.size < args.total:
    pad = rng.integers(1, args.hot_pool + 1, size=args.total - samples.size)
    samples = np.concatenate([samples, pad])
samples = (samples[:args.total] - 1).astype(np.int64)

# Pull queries
out = hot[samples]
# Stats
from collections import Counter
top = Counter(samples).most_common(10)
print(f"hot_pool={args.hot_pool} alpha={args.alpha} total={args.total}")
print(f"top-10 freq: {top}")
print(f"unique sampled: {len(set(samples.tolist()))}")

with open(args.out, "wb") as f:
    f.write(struct.pack("II", args.total, d))
    f.write(out.tobytes())
print(f"wrote {args.out}: {args.total} × {d} float32")
