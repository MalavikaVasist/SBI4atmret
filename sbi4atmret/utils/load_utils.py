import importlib


def load_callable(module_path, name):
    module = importlib.import_module(module_path)
    return getattr(module, name)
