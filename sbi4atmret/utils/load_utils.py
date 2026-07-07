"""
Dynamic import utilities for config-driven component instantiation.
"""

from importlib import import_module


def load_callable(dotted_path: str):
    """
    Import a class or function from a dotted module path string.

    Supports both absolute imports and sbi4atmret-relative imports:
        - "sbi4atmret.simulators.simulator.Simulator" (absolute)
        - "simulators.simulator.Simulator" (relative to sbi4atmret)

    Args:
        dotted_path: dotted path like "module.submodule.ClassName"

    Returns:
        The class or function object.
    """
    # Split into module path and attribute name
    parts = dotted_path.rsplit(".", 1)

    if len(parts) == 1:
        raise ImportError(f"Cannot import '{dotted_path}': need at least module.name")

    module_path, attr_name = parts

    # Try absolute import first
    try:
        module = import_module(module_path)
        return getattr(module, attr_name)
    except (ModuleNotFoundError, AttributeError):
        pass

    # Try prefixing with sbi4atmret
    prefixed = f"sbi4atmret.{module_path}"
    try:
        module = import_module(prefixed)
        return getattr(module, attr_name)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(
            f"Cannot import '{attr_name}' from '{module_path}' "
            f"(also tried '{prefixed}'): {e}"
        ) from e
