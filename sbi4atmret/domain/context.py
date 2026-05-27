from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class DomainContext:

    # core scientific components
    simulator_dict: Dict[str, Any]
    observation: Any

    # preprocessing / inference components
    pipe: Any
    noise: Any

    # domain metadata

    param_index: dict

    sim_wlens: dict
    obs_wlens: dict

    obs_noise: dict

    scale: float

    unsort_index: list


