import numpy as np
import sys


# Global pressure array for AMR mode
PGLOBAL = None


def initialize_pressure_layers(press, parameters):
    """
    Initialize pressure layers, optionally with Adaptive Mesh Refinement (AMR).

    Args:
        press: pressure array from atmosphere (in bar)
        parameters: dict of Parameter objects

    Returns:
        p_use: pressure array to use for PT profile computation
    """
    amr = parameters['AMR'].value if hasattr(parameters['AMR'], 'value') else parameters['AMR']

    if amr:
        try:
            pglobal_check(
                press,
                parameters['pressure_simple'].value,
                parameters['pressure_scaling'].value,
            )
        except KeyError:
            print("You must include the pressure_simple and pressure_scaling parameters when using AMR!")
            sys.exit(1)

        p_use = PGLOBAL  # len 1000 in bar
    else:
        p_use = press  # len 127 in bar

    return p_use


def pglobal_check(press, shape, scaling):
    """
    Check and update the global pressure array PGLOBAL.

    Args:
        press: pressure array from pRT_object (sets min/max)
        shape: pressure array length without AMR
        scaling: resolution scaling factor
    """
    global PGLOBAL

    target_size = int(scaling * shape)

    if PGLOBAL is None or PGLOBAL.shape[0] != target_size:
        PGLOBAL = np.logspace(
            np.log10(press[0]),
            np.log10(press[-1]),
            target_size,
        )
