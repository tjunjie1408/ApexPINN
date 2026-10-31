# ApexPINN

Physics-informed forecasting of Formula 1 tire degradation from public FastF1 data, written in JAX/Equinox,
feeding a vectorized JAX pit-strategy simulator.

> Work in progress. No results yet.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                     # CPU (Windows, macOS, Linux)
uv sync --extra cuda        # NVIDIA GPU on Linux / WSL2
uv run pytest               # fast test suite
uv run apexpinn-demo        # synthetic demo
```
