# Architecture notes

A map of how the pieces fit, which oddities are deliberate, and where
the known debt sits. Directory-level layout is in the README; this is
the reasoning layer.

## Three build roots, on purpose

    CMakeLists.txt                      torch-based library + benchmarks
    src/index/concurrent/CMakeLists.txt the concurrent engine
    tools/CMakeLists.txt                ground-truth utilities

They stay separate because their dependency worlds barely overlap: the
library needs Torch (and optionally CUDA, SPTAG, puck, faiss); the
engine needs TBB and pybind11 and nothing of Torch; the tools need a
compiler and, optionally, CUDA. One root would force every build to
configure all three stacks and inherit the library's aggressive global
flags (-Ofast, -march=native applied at directory scope). Three small
configures beat one entangled one. The cost is that nothing builds
"everything" — each guide states its own build line.

## Two index interfaces, one directory

src/index holds both generations: the torch-tensor families
(`CANDOR::AbstractIndex`, insert/search take tensors, configured via
ConfigMap) and src/index/concurrent (`IndexBase<float>`, raw pointers
plus tags, a visibility watermark for snapshot reads). They are not
unified and there is no adapter. That is a known seam, kept because
the APIs serve different masters: the tensor API exists for the
PyCANDOR/torch ecosystem, the pointer API exists so the benchmark
driver can run lock-step with raw buffers and no framework in the hot
path. Bridging them (an AbstractIndex shim over IndexBase) is doable
if a workload ever needs a concurrent-capable index behind the tensor
API; nobody has needed it yet.

The benchmark driver (driver/) and the pybind module (python/) live
inside src/index/concurrent rather than with the python harness. The
engine is one self-contained build unit — backends, driver, bindings
compile together against the same vendored hnswlib fork — and keeping
the unit whole won over keeping the harness whole. The python side of
that trade is src/concurrency: pure orchestration, no compiled parts.

## Python distribution model

There is no pyproject for the concurrency package; it is run in place
with PYTHONPATH=src. Deliberate for now: the repo already ships a
setup.py wired to the torch library's PyCANDOR module, and a second
packaging entry point at the root would fight it for pip's attention.
The cost is the PYTHONPATH prefix on every command and the sys.path
bootstrap in the test scripts. If the harness ever needs to be
installable, give it its own pyproject under src/concurrency and keep
the root for PyCANDOR.

## Source collection in the library build

The library gathers sources through add_sources/get_sources macros
(cmake/macros.cmake) accumulating into a global property that each
subdirectory CMakeLists feeds. It predates this refactor and works,
but file names live far from the add_library call, so a missing file
surfaces at generate time with a one-line error and no context. When a
build breaks right after moving files, look at the subdirectory
CMakeLists first.

## Tests

test/concurrency is the live suite — every python test runs directly
and in CI-able form. test/index holds the library's Catch2 system
tests; they had been disconnected from the build entirely (the
add_subdirectory was commented out, likely when they last broke).
They are now wired behind the existing ENABLE_UNIT_TESTS option.

## Frozen zones

scripts/ (the legacy experiment scripts), figures/, and the thirdparty
checkouts are read-only as far as conventions go: nothing in them is
renamed, restructured or linted. They are inputs to published results;
consistency sweeps stop at their borders.
