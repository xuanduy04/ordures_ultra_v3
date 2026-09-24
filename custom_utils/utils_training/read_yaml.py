import sys
import yaml


def get_config_value(config_path: str, dotted_key: str) -> str:
    with open(config_path) as f:
        config = yaml.safe_load(f)
     
    value = config
    prefix = []
    for key in dotted_key.strip().strip(".").split("."):
        try:
            value = value[key]
            prefix.append(key)
        except (KeyError, TypeError):
            parent_path = ".".join(prefix) if prefix else "root"
            raise KeyError(f"'{key}' not in {parent_path} (full key: '{dotted_key}')")
    return value


if __name__ == "__main__":
    try:
        print(get_config_value(sys.argv[1], sys.argv[2]))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
