import datetime as dt
import json

import numpy as np
import polars as pl

from model_orm import UsageParams

# Plug times: same offset applied to both so the drive-window length is preserved
PLUG_TIME_NOISE_STD_MINUTES = 45

# Continuous parameters, relative std deviation (fraction of archetype mean). These are template values that can be adjusted based in data driven research
KWH_PER_WEEK_NOISE_REL      = 1.0
AVG_DRIVES_PER_DAY_NOISE_REL = 1.5
BATTERY_KWH_NOISE_REL       = 0.10
MILES_PER_KWH_NOISE_REL     = 0.10

# Target SoC, absolute std deviation in fraction units
TARGET_SOC_NOISE_ABS = 0.1


def add_noise(rng: np.random.Generator, mean: float, rel_std: float, lo: float = 0.0, hi: float = float("inf")) -> float:
    """Sample mean x(1 + N(0, rel_std)), clamped to [lo, hi]."""
    return float(np.clip(mean * (1.0 + rng.normal(0.0, rel_std)), lo, hi))


def parse_time_to_minutes(time_str: str) -> int:
    """Convert a time string like '6:00 PM' or '12:00 AM' to minutes since midnight."""
    parsed = dt.datetime.strptime(time_str.strip(), "%I:%M %p")
    return parsed.hour * 60 + parsed.minute


def minutes_to_datetime(minutes: int) -> dt.datetime:
    """Convert minutes since midnight (0-1439) to a datetime on a fixed reference date."""
    minutes = minutes % (24 * 60)
    return dt.datetime(2024, 1, 1, minutes // 60, minutes % 60)


def parse_target_soc(soc_str: str) -> float:
    """Parse '80%' -> 0.80 (fractional SoC)."""
    return float(soc_str.strip().replace("%", "")) / 100.0


def sample_population(n_vehicles: int, seed: int) -> list[UsageParams]:
    """
    Sample n_vehicles agents from the archetype population distribution.

    Each agent is drawn from one of the archetypes, weighted by their
    population_percentage. Gaussian noise is applied to every continuous
    parameter so that agents within the same archetype are realistic
    individuals rather than identical copies.

    Noise strategy:
      - Plug times : shared additive offset in minutes (preserves drive-window duration)
      - kWh / week : relative noise (±20 %)
      - avg_drives/day : relative noise (±20 %), clamped > 0
      - charger kW : relative noise (±10 %), clamped > 0
      - battery kWh : relative noise (±10 %), clamped > 0
      - miles / kWh : relative noise (±10 %), clamped > 0
      - target SoC : absolute additive noise (±5 pp), clamped [0, 1]

    Results are written to data_store/simulation_inputs/usage_params.json
    and returned as a list of UsageParams.
    """
    rng = np.random.default_rng(seed)
    df = pl.read_csv("./data_store/model_data/usage_profiles.csv")

    # Build a normalised weight vector from population percentages
    weights = df["population_percentage"].to_numpy().astype(float)
    weights /= weights.sum()

    # Sample archetype row indices for all n_vehicles at once
    archetype_indices = rng.choice(len(df), size=n_vehicles, p=weights)

    agents: list[UsageParams] = []

    for i, idx in enumerate(archetype_indices):
        row = df.row(idx, named=True)

        plug_in_minutes  = parse_time_to_minutes(row["normal_plug_in_time"])
        plug_out_minutes = parse_time_to_minutes(row["normal_plug_out_time"])

        time_offset = int(round(rng.normal(0, PLUG_TIME_NOISE_STD_MINUTES)))
        plug_in_minutes_noisy  = (plug_in_minutes  + time_offset) % (24 * 60)
        plug_out_minutes_noisy = (plug_out_minutes + time_offset) % (24 * 60)

        base_soc = parse_target_soc(row["target_soc_for_charging"])

        agent = UsageParams(
            vehicle_name=f"vehicle_{i}",
            kWh_per_week=add_noise(rng, row["kWh_per_week"], KWH_PER_WEEK_NOISE_REL, lo=0.1),
            plug_in_frequency=row["plug_in_frequency"],
            avg_drives_per_day=add_noise(rng, row["avg_drives_per_day"], AVG_DRIVES_PER_DAY_NOISE_REL, lo=0.01),
            charger_kW=row["charger_kW"],
            normal_plug_in_time=minutes_to_datetime(plug_in_minutes_noisy),
            normal_plug_out_time=minutes_to_datetime(plug_out_minutes_noisy),
            battery_kWh=add_noise(rng, row["battery_kWh"], BATTERY_KWH_NOISE_REL, lo=1.0),
            miles_per_kWh=add_noise(rng, row["miles_per_kWh"], MILES_PER_KWH_NOISE_REL, lo=0.1),
            target_soc_for_charging=add_noise(rng, base_soc, TARGET_SOC_NOISE_ABS, lo=0.0, hi=1.0),
        )
        agents.append(agent)

    with open("./data_store/simulation_inputs/usage_params.json", "w") as profiles:
        json.dump(
            [agent.model_dump(mode="json") for agent in agents],
            profiles,
            indent=2,
        )

    return agents
