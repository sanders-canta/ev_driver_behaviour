import datetime as dt
from pydantic import BaseModel, Field

class UsageParams(BaseModel):
    """This orm will serve as the input for one agent of the simulation."""
    vehicle_name: str = Field(description="Unique identifier for this agent.")
    kWh_per_week: float = Field(ge=0, description="Average energy consumed per week in kWh. Equivalent to weekly mileage once divided by efficiency.")
    plug_in_frequency: float = Field(ge=0, le=1, description="Probability of plugging in on any given day (0 to 1). E.g. 1.0 = every day, 0.2 = roughly once every 5 days.")
    avg_drives_per_day: float = Field(ge=0, description="Average number of drive sessions per day. Values < 1 mean less-than-daily driving (e.g. 0.2 = one drive every 5 days). Values >= 1 mean multiple sessions per day (e.g. 2.0 = morning and evening commute). Determines both drive frequency and session duration.")
    charger_kW: float = Field(ge=0, description="Rated power of the home charger in kW. Assumed to always charge at this maximum rate.")
    normal_plug_in_time: dt.datetime = Field(description="Typical time of day the driver arrives home and plugs in. Only the hour and minute components are used by the simulator.")
    normal_plug_out_time: dt.datetime = Field(description="Typical time of day the driver leaves home and unplugs. Only the hour and minute components are used by the simulator.")
    battery_kWh: float = Field(ge=0, description="Usable battery capacity of the vehicle in kWh. Determines how quickly SoC changes per unit of energy.")
    miles_per_kWh: float = Field(ge=0, description="Vehicle efficiency in miles per kWh. Used to convert energy consumption to distance if needed.")
    target_soc_for_charging: float = Field(ge=0, le=1, description=f"SoC fraction (0 to 1) at which the charger stops, e.g. 0.8 for an 80 % charge limit.")