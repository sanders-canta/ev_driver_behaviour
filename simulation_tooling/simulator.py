import datetime as dt

import numpy as np
import polars as pl

from model_orm import UsageParams

MINUTES_PER_DAY = 1440
DAYS_PER_WEEK = 7
TOTAL_MINUTES = MINUTES_PER_DAY * DAYS_PER_WEEK


def build_schedule(
    plug_out_minute: int,
    plug_in_minute: int,
    drive_session_minutes: int,
    avg_drives_per_day: float,
    plug_in_frequency: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build plugged_in and driving boolean arrays for the whole week.

    The drive window is defined as [plug_out_minute, plug_in_minute).
    On a driving day, n_drives sessions of drive_session_minutes are spaced
    evenly within the window with equal rest gaps between them. Minutes in the
    window that are not part of a drive session are resting (neither plugged in
    nor driving). Minutes outside the window are plugged in (charging).

    avg_drives_per_day < 1 - Bernoulli trial each day; 0 or 1 session.
    avg_drives_per_day >= 1 - every day is a drive day; n sessions determined by floor + Bernoulli on the fractional part.

    Returns:
        plugged_in : True when the car is connected and can charge.
        driving : True when the car is actively being driven.
    """
    plugged_in = np.ones(TOTAL_MINUTES, dtype=bool)
    driving = np.zeros(TOTAL_MINUTES, dtype=bool)

    # Drive window length in minutes (handles wrap-around midnight)
    if plug_out_minute < plug_in_minute:
        window_minutes = plug_in_minute - plug_out_minute
    else:
        window_minutes = MINUTES_PER_DAY - plug_out_minute + plug_in_minute
    window_minutes = max(window_minutes, 1)

    for day in range(DAYS_PER_WEEK):
        plugged_in_bool = 1 if rng.random() < plug_in_frequency else 0
        # Determine number of drive sessions for this day
        # Add per-day Gaussian noise so each day has a slightly different effective rate
        daily_avg = max(0.01, avg_drives_per_day + rng.normal(0, avg_drives_per_day * 0.3))
        if daily_avg < 1:
            n_drives = 1 if rng.random() < daily_avg else 0
        else:
            floor_d = int(daily_avg)
            extra = 1 if rng.random() < (daily_avg - floor_d) else 0
            n_drives = floor_d + extra

        day_offset = day * MINUTES_PER_DAY

        if not plugged_in_bool:
            plugged_in[day_offset:day_offset + MINUTES_PER_DAY] = False

        if n_drives == 0:
            continue  # non-driving day: no drive window to unplug

        # Cap each session so all n_drives fit inside the window
        session_mins = max(1, min(drive_session_minutes, window_minutes // n_drives))
        total_drive_mins = n_drives * session_mins
        rest_mins = window_minutes - total_drive_mins
        gap = rest_mins / (n_drives + 1)  # equal spacing between sessions

        # Unplug the entire drive window (rest and drive periods)
        for offset in range(window_minutes):
            gm = day_offset + plug_out_minute + offset
            if 0 <= gm < TOTAL_MINUTES:
                plugged_in[gm] = False

        # Mark individual drive sessions within the window
        for i in range(n_drives):
            session_start = int(gap * (i + 1)) + session_mins * i
            for j in range(session_mins):
                gm = day_offset + plug_out_minute + session_start + j
                if 0 <= gm < TOTAL_MINUTES:
                    driving[gm] = True
    
    return plugged_in, driving


def simulate_soc(
    plugged_in: np.ndarray,
    driving: np.ndarray,
    soc_start: float,
    target_soc: float,
    soc_drain_per_minute: np.ndarray,
    soc_charge_per_minute: float,
) -> np.ndarray:
    """Simulate the state-of-charge (SoC) time series for one agent over one week."""
    soc = np.empty(TOTAL_MINUTES, dtype=np.float64)
    current = soc_start

    for t in range(TOTAL_MINUTES):
        soc[t] = current
        if driving[t]:
            current = max(0.0, current - soc_drain_per_minute[t])
        elif plugged_in[t]:
            current = min(target_soc, current + soc_charge_per_minute)

    return soc


def simulate_agent(agent: UsageParams, rng: np.random.Generator) -> pl.DataFrame:
    """Run a one-week SoC simulation for a single agent and return a Polars DataFrame."""
    plug_in_minute  = agent.normal_plug_in_time.hour  * 60 + agent.normal_plug_in_time.minute
    plug_out_minute = agent.normal_plug_out_time.hour * 60 + agent.normal_plug_out_time.minute

    # Energy consumed per individual drive session
    energy_per_drive_kWh = agent.kWh_per_week / 7 / agent.avg_drives_per_day

    # Drive session duration: energy / charger power gives hours, × 60 → minutes
    # charger_kW is used as a proxy for average vehicle power draw while driving
    drive_session_minutes = max(1, int(energy_per_drive_kWh / agent.charger_kW * 60))

    # Base SoC fraction drained per minute during a drive session
    base_soc_drain_per_minute = (energy_per_drive_kWh / agent.battery_kWh) / drive_session_minutes

    # SoC fraction gained per minute while plugged in
    soc_charge_per_minute = (agent.charger_kW / 60.0) / agent.battery_kWh

    plugged_in, driving = build_schedule(
        plug_out_minute=plug_out_minute,
        plug_in_minute=plug_in_minute,
        drive_session_minutes=drive_session_minutes,
        avg_drives_per_day=agent.avg_drives_per_day,
        plug_in_frequency=agent.plug_in_frequency,
        rng=rng,
    )

    # Build per-minute drain array with ±60% per-day variation in drive distance to make not all drives the same
    soc_drain_per_minute = np.zeros(TOTAL_MINUTES, dtype=np.float64)
    for day in range(DAYS_PER_WEEK):
        day_scale = max(0.1, rng.normal(1.0, 0.6))
        start = day * MINUTES_PER_DAY
        end = (day + 1) * MINUTES_PER_DAY
        soc_drain_per_minute[start:end] = driving[start:end] * base_soc_drain_per_minute * day_scale

    # Sim starts Monday 00:00 at target_soc (car is freshly charged at week start)
    soc = simulate_soc(
        plugged_in=plugged_in,
        driving=driving,
        soc_start=agent.target_soc_for_charging,
        target_soc=agent.target_soc_for_charging,
        soc_drain_per_minute=soc_drain_per_minute,
        soc_charge_per_minute=soc_charge_per_minute,
    )

    # Build timestamp index
    start_dt = dt.datetime(2026, 1, 5, 0, 0)
    timestamps = pl.datetime_range(
        start=start_dt,
        end=start_dt + dt.timedelta(minutes=TOTAL_MINUTES - 1),
        interval="1m",
        eager=True,
    )

    return pl.DataFrame({
        "timestamp":    timestamps,
        "soc":          soc,
        "plugged_in":   plugged_in,
        "driving":      driving,
        "vehicle_name": agent.vehicle_name,
    })


def simulate(agents: list[UsageParams], seed: int) -> None:
    """Simulate one week of SoC for each agent and write results to individual parquet files in data_store/simulation_results/{vehicle_name}.parquet."""
    rng = np.random.default_rng(seed)

    for agent in agents:
        df = simulate_agent(agent, rng)
        out_path = f"./data_store/simulation_results/{agent.vehicle_name}.parquet"
        df.write_parquet(out_path)