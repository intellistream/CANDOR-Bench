#!/usr/bin/env python3
"""Generate a locality query stream: queries correlated with insert vectors.

Each query = random insert vector (from the 2nd half of the dataset) + gaussian noise.
This simulates a real-world workload where searches are related to recent inserts
(e.g., RAG: query about recently ingested document; recommender: search near new items).

Output: binary file compatible with the benchmark's bin format (npts, dim, float32 data).
"""

import numpy as np
import struct
import sys

def read_bin(path):
    with open(path, 'rb') as f:
        npts = struct.unpack('I', f.read(4))[0]
        dim = struct.unpack('I', f.read(4))[0]
        data = np.frombuffer(f.read(npts * dim * 4), dtype=np.float32).reshape(npts, dim)
    return data

def write_bin(path, data):
    npts, dim = data.shape
    with open(path, 'wb') as f:
        f.write(struct.pack('I', npts))
        f.write(struct.pack('I', dim))
        f.write(data.astype(np.float32).tobytes())

def main():
    base_path = sys.argv[1]  # e.g., ../data/sift/sift_base.bin
    out_path = sys.argv[2]   # e.g., ../data/sift/sift_query_locality.bin
    num_queries = int(sys.argv[3]) if len(sys.argv) > 3 else 1000000
    noise_scale = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
    begin_num = int(sys.argv[5]) if len(sys.argv) > 5 else 500000

    print(f"Reading base vectors from {base_path}...")
    base = read_bin(base_path)
    npts, dim = base.shape
    print(f"Base: {npts} points, {dim} dims")

    # Insert vectors are from begin_num to npts
    insert_vecs = base[begin_num:]
    n_insert = len(insert_vecs)
    print(f"Insert vectors: {n_insert} (indices {begin_num}-{npts})")

    # Compute noise scale relative to data magnitude
    avg_norm = np.mean(np.linalg.norm(insert_vecs, axis=1))
    actual_noise = noise_scale * avg_norm
    print(f"Avg vector norm: {avg_norm:.1f}, noise scale: {actual_noise:.1f} ({noise_scale*100:.0f}%)")

    # Generate queries: pick random insert vector + noise
    rng = np.random.default_rng(42)
    indices = rng.integers(0, n_insert, size=num_queries)
    noise = rng.normal(0, actual_noise, size=(num_queries, dim)).astype(np.float32)
    queries = insert_vecs[indices] + noise

    write_bin(out_path, queries)
    print(f"Written {num_queries} locality queries to {out_path}")

if __name__ == '__main__':
    main()
