from pathlib import Path
import yaml


class Config:
    def __init__(self, config_path):
        self.path = Path(config_path)

        if not self.path.exists():
            raise FileNotFoundError(
                f"Config not found: {self.path}\n"
                f"CWD: {Path.cwd()}"
            )

        with open(self.path, "r") as f:
            self.cfg = yaml.safe_load(f)

    def get(self, key, default=None):
        return self.cfg.get(key, default)

    def section(self, name):
        return self.cfg.get(name, {})

    def __getitem__(self, key):
        return self.cfg[key]