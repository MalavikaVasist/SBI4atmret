"""
Configuration selection utilities for handling multi-model configs.
"""

from typing import Dict, Any, List, Tuple, Union


def _select_by_index(value: Any, index: int) -> Any:
    """Select the i-th element from a list/tuple, or return the value if not a sequence."""
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def _select_indexed_dict(config_dict: Dict[str, Any], i: int) -> Dict[str, Any]:
    """Recursively select indexed values from a nested dictionary."""
    selected = {}
    for key, value in config_dict.items():
        if isinstance(value, dict):
            selected[key] = _select_indexed_dict(value, i)
        elif isinstance(value, (list, tuple)) and len(value) > i and not isinstance(value[0], dict):
            selected[key] = _select_by_index(value, i)
        else:
            selected[key] = value
    return selected


def _select_model_config(model_config: Dict[str, Any], i: int) -> Dict[str, Any]:
    """Select model configuration for the i-th index."""
    selected = {
        'model': model_config.get('model'),
        'model_name': _select_by_index(model_config.get('model_name') or model_config.get('name'), i),
        'embedding': {
            'miri': _select_by_index(model_config['embedding']['miri'], i),
            'gemini': _select_by_index(model_config['embedding']['gemini'], i),
            'miri_output': _select_by_index(model_config['embedding']['miri_output'], i),
            'gemini_output': _select_by_index(model_config['embedding']['gemini_output'], i),
        },
        'estimator': _select_indexed_dict(model_config['estimator'], i) if model_config.get('estimator') else None,
        'hidden_features': _select_by_index(model_config.get('hidden_features'), i),
        'no_of_params': _select_by_index(model_config.get('no_of_params'), i),
        'transforms': _select_by_index(model_config.get('transforms'), i),
        'signal': _select_by_index(model_config.get('signal'), i),
    }
    if 'batch_size' in model_config:
        selected['batch_size'] = _select_by_index(model_config['batch_size'], i)
    return selected


def _select_loss_config(loss_config: Dict[str, Any], i: int) -> Dict[str, Any]:
    """Select loss configuration for the i-th index."""
    selected = {
        'loss_type': _select_by_index(loss_config.get('loss_type'), i),
        'optimizer': _select_indexed_dict(loss_config.get('optimizer', {}), i) if loss_config.get('optimizer') else None,
        'scheduler': _select_indexed_dict(loss_config.get('scheduler', {}), i) if loss_config.get('scheduler') else None,
    }
    return selected


def _select_training_config(training_config: Dict[str, Any], i: int) -> Dict[str, Any]:
    """Select training configuration for the i-th index."""
    selected = dict(training_config)
    for key in ['epochs', 'epoch_fin', 'batch_size']:
        if key in training_config:
            selected[key] = _select_by_index(training_config[key], i)
    if 'optimizer' in training_config:
        selected['optimizer'] = _select_indexed_dict(training_config['optimizer'], i)
    if 'scheduler' in training_config:
        selected['scheduler'] = _select_indexed_dict(training_config['scheduler'], i)
    return selected


def select_index_config(full_config: Dict[str, Any], i: int) -> Dict[str, Any]:
    """
    Create a per-index configuration snapshot from the full config.

    Args:
        full_config: The full configuration dictionary.
        i: The model index to select from array-valued config fields.

    Returns:
        dict: A config dictionary where indexed values are reduced to the i'th element.
    """
    selected = dict(full_config)
    selected['ML_model_configs'] = _select_model_config(full_config['ML_model_configs'], i)
    if full_config.get('Loss') is not None:
        selected['Loss'] = _select_loss_config(full_config['Loss'], i)
    if full_config.get('training') is not None:
        selected['training'] = _select_training_config(full_config['training'], i)
    return selected
