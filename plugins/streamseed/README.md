# streamseed

`streamseed` is a standalone Python package for StreamSeed-Core query-hint optimization in dynamic ANNS.

## API (minimal)

- `StreamSeedPlugin`: backend-agnostic plugin orchestration class.
- `FaissHnswStreamSeedBackend`: Faiss incremental HNSW backend implementation.
- `configure(...)`: configure StreamSeed-Core behavior before query.
- `query(...)`: calls `PyCANDYAlgo.IndexFAISS.search_warm` with StreamSeed-Core parameters.

## Core layout

- C++ StreamSeed-Core now lives in `plugins/streamseed/core`:
	- `include/streamseed/StreamSeedCore.h`
	- `src/StreamSeedCore.cpp`
- `algorithms_impl/faiss/faiss/StreamSeedCore.h` is kept as a compatibility forwarding header.
- `algorithms_impl/faiss/faiss/CMakeLists.txt` compiles StreamSeed-Core from plugin path and injects it into `faiss` targets.

## Install (local)

```bash
cd plugins/streamseed
pip install -e .
```

## Notes

- This package expects `PyCANDYAlgo` to be available in the Python environment.
- It is intentionally decoupled from `bench.runner` and `BaseStreamingANN` to support PyPI publication.
