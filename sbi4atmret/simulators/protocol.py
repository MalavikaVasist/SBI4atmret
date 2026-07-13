"""
Simulator Protocol — the contract any simulator must satisfy.

Any class that conforms to this protocol can be used as a simulator
in SBI4atmret without inheriting from a specific base class.

Usage:
    To create a new simulator, implement a class with:
    - `names`: list of parameter names (column order in theta)
    - `wavelength`: 1-D array of output wavelengths in microns
    - `scale`: float scaling factor applied to raw spectra
    - `__call__(theta) -> SimulatorOutput`: the forward model

Example:
    class MySimulator:
        names = ["param_a", "param_b", "param_c"]
        scale = 1.0

        def __init__(self, a, b, **kwargs):
            ...
            self.wavelength = np.linspace(1.0, 5.0, 500)

        def __call__(self, theta):
            spectrum = my_forward_model(theta)
            return SimulatorOutput(
                wavelength=self.wavelength,
                spectrum=spectrum,
            )

    Then in config YAML:
        simulator_config:
          my_instrument:
            type: my_package.my_module.MySimulator
            kwargs:
              a: 1.0
              b: 2.0
              names: [param_a, param_b, param_c]
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Protocol, runtime_checkable

import numpy as np


# =========================================================
# STRUCTURED OUTPUT
# =========================================================

@dataclass(frozen=True)
class SimulatorOutput:
    """
    Standardized simulator output container.

    Required fields:
        wavelength: (D,) array in microns
        spectrum: (D,) array of flux values (after scaling)

    Optional fields (used by PT plotting, contribution analysis):
        contribution: (n_pressures, D) emission contribution function
        pressures: (n_pressures,) pressure grid in bar
        temperatures: (n_pressures,) temperature profile in K
        metadata: free-form dict for debugging
        parameters: dict of {name: value} used in this call
    """

    wavelength: np.ndarray
    spectrum: np.ndarray

    contribution: Optional[np.ndarray] = None
    pressures: Optional[np.ndarray] = None
    temperatures: Optional[np.ndarray] = None

    metadata: Optional[Dict[str, Any]] = None
    parameters: Optional[Dict[str, Any]] = None


# =========================================================
# PROTOCOL
# =========================================================

@runtime_checkable
class SimulatorProtocol(Protocol):
    """
    Protocol that all simulators must satisfy.

    The config system instantiates simulators via:
        simulator = load_callable(type)(**kwargs)

    At runtime, the framework accesses:
        simulator.names       — to build theta_mapper indices
        simulator.wavelength  — to build domain wavelength grids
        simulator.scale       — for noise model scaling
        simulator(theta)      — to generate spectra

    Type checking:
        isinstance(my_sim, SimulatorProtocol)  # True if it conforms
    """

    @property
    def names(self) -> List[str]:
        """
        Parameter names in the order they appear as columns in theta.

        Must match the `names` list in the simulator_config YAML.
        The theta_mapper uses these to build split/merge index tables.
        """
        ...

    @property
    def wavelength(self) -> np.ndarray:
        """
        Output wavelength grid in microns, shape (D,).

        Set once at initialization. Used by:
        - DomainContext (sim_wlens)
        - Pipe mask construction
        - Bolometric integration
        """
        ...

    @property
    def scale(self) -> float:
        """
        Multiplicative scaling applied to raw spectra.

        Used by the noise model: error = sigma * randn * scale.
        Set to 1.0 if spectra are already in final units.
        """
        ...

    def __call__(self, theta: np.ndarray) -> SimulatorOutput:
        """
        Run the forward model.

        Args:
            theta: 1-D array of shape (n_params,) in the order of self.names.

        Returns:
            SimulatorOutput with at minimum:
                - wavelength: same as self.wavelength
                - spectrum: (D,) flux array (already scaled by self.scale)

            Optional (for PT/contribution diagnostics):
                - pressures, temperatures, contribution
        """
        ...
