from dataclasses import dataclass


@dataclass(frozen=True)
class PowerState:
    battery_percent: int | None
    battery_seconds: int | None
    on_battery: bool
    no_battery: bool
    charging: bool
    charged: bool
    plugged_in: bool
    has_battery: bool
