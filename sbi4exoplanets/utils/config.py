import yaml
from pathlib import Path


class Config:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        with open(self.config_path, 'r') as f:
            self.data = yaml.safe_load(f)

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)