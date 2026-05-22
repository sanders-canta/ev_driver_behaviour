"""
FastAPI backend for the EV Driver Behaviour Simulator dashboard.

Run from the repo root:
    uvicorn dashboard.api:app --reload
"""

import asyncio
import datetime as dt
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Make simulation_tooling importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent / "simulation_tooling"))

from constants import seed  # noqa: E402
from model_orm import UsageParams  # noqa: E402
from population_distribution import sample_population  # noqa: E402
from simulator import simulate  # noqa: E402

app = FastAPI(title="EV Driver Behaviour Simulator")

_BASE_DIR = Path(__file__).parent.parent
_PROFILES_PATH = _BASE_DIR / "data_store" / "simulation_inputs" / "usage_params.json"
_STATIC_DIR = Path(__file__).parent / "static"

# One worker so simulations are serialised and don't fight over files
_executor = ThreadPoolExecutor(max_workers=1)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    return (_STATIC_DIR / "index.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------

def _load_profiles() -> list[dict]:
    if not _PROFILES_PATH.exists():
        return []
    return json.loads(_PROFILES_PATH.read_text())


def _save_profiles(profiles: list[dict]) -> None:
    _PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PROFILES_PATH.write_text(json.dumps(profiles, indent=2, default=str))


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/profiles")
async def get_profiles():
    """Return all currently saved usage profiles."""
    return _load_profiles()


class GenerateRequest(BaseModel):
    n_vehicles: int
    seed: int = seed


@app.post("/api/profiles/generate")
async def generate_profiles(req: GenerateRequest):
    """
    Sample n_vehicles agents from the archetype population distribution
    and save them, replacing any existing profiles.
    """
    if req.n_vehicles < 1:
        raise HTTPException(400, "n_vehicles must be at least 1")

    loop = asyncio.get_event_loop()
    agents = await loop.run_in_executor(
        _executor,
        lambda: sample_population(n_vehicles=req.n_vehicles, seed=req.seed),
    )
    return {"message": f"Generated {len(agents)} profiles", "count": len(agents)}


class AddProfileRequest(BaseModel):
    vehicle_name: str
    kWh_per_week: float
    plug_in_frequency: float
    avg_drives_per_day: float
    charger_kW: float
    normal_plug_in_time: str   # "HH:MM" from the HTML time input
    normal_plug_out_time: str  # "HH:MM"
    battery_kWh: float
    miles_per_kWh: float
    target_soc_for_charging: float  # fraction 0-1


@app.post("/api/profiles/add")
async def add_profile(req: AddProfileRequest):
    """Manually append one profile to the inputs file."""
    def _parse_time(t: str) -> dt.datetime:
        h, m = map(int, t.split(":"))
        return dt.datetime(2024, 1, 1, h, m)

    try:
        agent = UsageParams(
            vehicle_name=req.vehicle_name,
            kWh_per_week=req.kWh_per_week,
            plug_in_frequency=req.plug_in_frequency,
            avg_drives_per_day=req.avg_drives_per_day,
            charger_kW=req.charger_kW,
            normal_plug_in_time=_parse_time(req.normal_plug_in_time),
            normal_plug_out_time=_parse_time(req.normal_plug_out_time),
            battery_kWh=req.battery_kWh,
            miles_per_kWh=req.miles_per_kWh,
            target_soc_for_charging=req.target_soc_for_charging,
        )
    except Exception as e:
        raise HTTPException(400, str(e))

    profiles = _load_profiles()
    if any(p["vehicle_name"] == req.vehicle_name for p in profiles):
        raise HTTPException(409, f"Profile '{req.vehicle_name}' already exists")

    profiles.append(agent.model_dump(mode="json"))
    _save_profiles(profiles)
    return {"message": f"Added profile '{req.vehicle_name}'"}


@app.delete("/api/profiles/{vehicle_name}")
async def delete_profile(vehicle_name: str):
    """Remove a profile by vehicle name."""
    profiles = _load_profiles()
    filtered = [p for p in profiles if p["vehicle_name"] != vehicle_name]
    if len(filtered) == len(profiles):
        raise HTTPException(404, f"Profile '{vehicle_name}' not found")
    _save_profiles(filtered)
    return {"message": f"Deleted '{vehicle_name}'"}


@app.post("/api/simulate")
async def run_simulation():
    """Run the SoC simulation for all active profiles."""
    profiles = _load_profiles()
    if not profiles:
        raise HTTPException(400, "No profiles to simulate — add some first")

    agents = [UsageParams.model_validate(p) for p in profiles]

    # Run in a thread so the event loop stays free; await completion before responding
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, lambda: simulate(agents, seed=seed))

    return {"message": f"Done — {len(agents)} agents written to data_store/simulation_results/"}


@app.post("/api/clear")
async def clear_simulation():
    """Delete all files in simulation_inputs and simulation_results."""
    inputs_dir  = _BASE_DIR / "data_store" / "simulation_inputs"
    results_dir = _BASE_DIR / "data_store" / "simulation_results"

    removed = 0
    for folder in (inputs_dir, results_dir):
        if folder.exists():
            for f in folder.iterdir():
                if f.is_file():
                    f.unlink()
                    removed += 1

    return {"message": f"Cleared {removed} file(s) from simulation_inputs and simulation_results"}


# ---------------------------------------------------------------------------
# Results API
# ---------------------------------------------------------------------------

_RESULTS_DIR = _BASE_DIR / "data_store" / "simulation_results"


@app.get("/api/results/vehicles")
async def list_result_vehicles():
    """Return sorted list of vehicle names that have simulation results."""
    if not _RESULTS_DIR.exists():
        return []
    return sorted(f.stem for f in _RESULTS_DIR.glob("*.parquet"))


@app.get("/api/results/vehicle/{vehicle_name}")
async def get_vehicle_result(vehicle_name: str):
    """Return downsampled SoC + plugged_in time series for one vehicle."""
    path = _RESULTS_DIR / f"{vehicle_name}.parquet"
    if not path.exists():
        raise HTTPException(404, f"No results for '{vehicle_name}'")

    loop = asyncio.get_event_loop()

    def _read():
        df = pl.read_parquet(path).gather_every(15)
        return {
            "labels":     df["timestamp"].dt.strftime("%a %H:%M").to_list(),
            "soc":        [round(v, 4) for v in df["soc"].to_list()],
            "plugged_in": df["plugged_in"].to_list(),
            "driving":    df["driving"].to_list(),
        }

    return await loop.run_in_executor(_executor, _read)


@app.get("/api/results/population")
async def get_population_result():
    """Return fleet-level stats: % plugged in, mean SoC, 95th-pct SoC."""
    files = sorted(_RESULTS_DIR.glob("*.parquet")) if _RESULTS_DIR.exists() else []
    if not files:
        raise HTTPException(404, "No simulation results found — run the simulation first")

    loop = asyncio.get_event_loop()

    def _compute():
        combined = pl.concat([pl.read_parquet(f) for f in files])
        agg = (
            combined
            .group_by("timestamp")
            .agg([
                (pl.col("plugged_in").cast(pl.Float64).mean() * 100).alias("pct_plugged_in"),
                pl.col("soc").mean().alias("mean_soc"),
                pl.col("soc").quantile(0.95).alias("p95_soc"),
                pl.col("soc").quantile(0.05).alias("p05_soc"),
            ])
            .sort("timestamp")
            .gather_every(15)
        )
        return {
            "labels":         agg["timestamp"].dt.strftime("%a %H:%M").to_list(),
            "pct_plugged_in": [round(v, 2) for v in agg["pct_plugged_in"].to_list()],
            "mean_soc":       [round(v, 4) for v in agg["mean_soc"].to_list()],
            "p95_soc":        [round(v, 4) for v in agg["p95_soc"].to_list()],
            "p05_soc":        [round(v, 4) for v in agg["p05_soc"].to_list()],
        }

    return await loop.run_in_executor(_executor, _compute)