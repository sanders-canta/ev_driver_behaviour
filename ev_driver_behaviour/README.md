# EV Driver Behaviour Simulator

An agent-based simulator of EV driver behaviour. Each agent is drawn from one of six UK driver archetypes and simulated at one-minute resolution over a full week, tracking battery state-of-charge and plug-in status. Results are visualised through a FastAPI dashboard.

## Setup

Sync dependencies using [UV](https://docs.astral.sh/uv/).
```bash
uv sync
```

Start the server from the repo root:

```bash
uv run uvicorn dashboard.api:app --reload
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

## Project structure

```
ev_driver_behaviour/
├── dashboard/
│   ├── api.py               # FastAPI backend
│   └── static/
│       └── index.html       # Single-page frontend
├── simulation_tooling/
│   ├── constants.py         # Random seed
│   ├── model_orm.py         # UsageParams Pydantic model
│   ├── population_distribution.py  # Archetype sampling
│   ├── simulator.py         # Minute-resolution SoC simulation
├── data_store/
│   ├── model_data/          # Archetype CSV (source usage data)
│   ├── simulation_inputs/   # Generated usage_params.json
│   └── simulation_results/  # Per-agent parquet output
└── pyproject.toml           # Dependency
```