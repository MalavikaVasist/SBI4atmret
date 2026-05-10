from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DomainContext:

    simulators: dict
    observation: Any

    param_index: dict

    sim_wlens: dict
    obs_wlens: dict

    obs_noise: dict

    scale: float

    unsort_index: list