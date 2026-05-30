#!/usr/bin/env python3
"""Compute brute-force KNN ground truth using faiss."""
import struct, sys, time
import numpy as np
import faiss

def read_bin(path):
    with open(path, 'rb') as f:
        n, d = struct.unpack('ii', f.read(8))
        data = np.frombuffer(f.read(n * d * 4), dtype=np.float32).reshape(n, d)
    print(f"Loaded {path}: {n} x {d}")
    return data

def write_gt(path, ids, dists, nq, k):
    with open(path, 'wb') as f:
        f.write(struct.pack('ii', nq, k))
        f.write(ids.astype(np.int32).tobytes())
        f.write(dists.astype(np.float32).tobytes())
    print(f"Wrote {path}")

def main():
    base_path = sys.argv[1]
    query_path = sys.argv[2]
    gt_path = sys.argv[3]
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 20

    base = read_bin(base_path)
    query = read_bin(query_path)
    
    n, d = base.shape
    nq = query.shape[0]
    print(f"Computing {k}-NN for {nq} queries against {n} base vectors (d={d})")
    
    t0 = time.time()
    index = faiss.IndexFlatL2(d)
    # Use all available threads
    faiss.omp_set_num_threads(faiss.omp_get_max_threads())
    print(f"Using {faiss.omp_get_max_threads()} threads")
    index.add(base)
    print(f"Index built in {time.time()-t0:.1f}s")
    
    t1 = time.time()
    dists, ids = index.search(query, k)
    print(f"Search done in {time.time()-t1:.1f}s")
    
    write_gt(gt_path, ids, dists, nq, k)
    print(f"Total time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
