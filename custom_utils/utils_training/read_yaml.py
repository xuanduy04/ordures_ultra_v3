import os
import sys
import yaml


def get_config_value(config_path: str, dotted_key: str) -> str:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    value = config
    for key in dotted_key.strip().strip(".").split("."):
        try:
            value = value[key]
        except (KeyError, TypeError):
            default = config.get("defaults")
            if not default:
                raise KeyError(f"'{dotted_key}' not in '{config_path}'")
            return get_config_value(os.path.join(os.path.dirname(config_path), default), dotted_key)
    return value


if __name__ == "__main__":
    try:
        print(get_config_value(sys.argv[1], sys.argv[2]))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
