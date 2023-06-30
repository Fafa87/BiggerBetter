import json
import pathlib

import attrdict

config_dir = pathlib.Path(__name__).parent.parent / "configs"


def load_config(name):
    config_path = config_dir / (name + ".json")
    with open(config_path) as f:
        return json.load(f)
