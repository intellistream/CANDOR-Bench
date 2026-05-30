# ANNchor M3

This directory is reset to a clean fork of the current `annchor-m2`
implementation. Failed M3 mechanism prototypes have been removed from the
algorithm path so the next attempt starts from the M2 baseline.

Removed mechanisms:

- insert construction admission / pause
- insert vector shadow and readset shadow
- insert vector NTA / cache hint experiments
- insert vector quarantine / micro-cache
- L0 RCU snapshot reader
- split visited-list pool / insert TLS visited-list probes
- search working-set trace hook

The `annchor-m3` CGO entry points are still present for compatibility with the
Go benchmark harness. The old admission setter is now a no-op stub and should
not be used as a mechanism.

Current boundary:

```text
M3 == M2 baseline + m3 symbol names only.
No paper-facing mechanism is active in this fork.
```
