from ..utils.config import Config
from ..utils.ees_general import Simulator
from ..utils.load_utils import load_callable


def build_simulator(config: Config):
    simulator_cfg = config["simulator"].copy()
    callable_cfg = simulator_cfg.pop("callable")

    emission_model = load_callable(
        callable_cfg["emission_model"]["module"],
        callable_cfg["emission_model"]["function"]
    )
    pt_profile = load_callable(
        callable_cfg["PT_profile"]["module"],
        callable_cfg["PT_profile"]["function"]
    )

    wav = config['wav']
    simulator = {}
    sim_type = simulator_cfg.pop('type')
    for atm_type in sim_type:
        simulator[atm_type] = {}
        for i, instrument in enumerate(config['instruments']):
            simulator[atm_type][instrument] = Simulator(
                emission_model_diseq=emission_model,
                PTprofile=pt_profile,
                a=wav[i][0],
                b=wav[i][1],
                **simulator_cfg
            )
    return simulator