import yaml

with open("parameters_5_yaml.yaml", "r") as f:
    config = yaml.safe_load(f)

# Flatten nested config and inject into globals
for section in config.values():
    if isinstance(section, dict):
        globals().update(section)
    else:
        # Top-level scalar values if any
        globals().update(config)
