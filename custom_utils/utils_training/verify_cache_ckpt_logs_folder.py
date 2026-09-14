import sys
from pathlib import Path

from read_yaml import get_config_value


if __name__ == "__main__":
    try:
        config_path = sys.argv[1]
        script_dir = sys.argv[2]

        print(f"{config_path=}\n{script_dir=}\n")
        
        checkpoint_dir = get_config_value(config_path, "checkpointing.checkpoint_dir")
        log_dir = get_config_value(config_path, "logger.log_dir")

        # Resolve absolute paths to ensure accurate comparison
        abs_script_dir = Path(script_dir).absolute().resolve()
        abs_checkpoint_parent = Path(checkpoint_dir).parent.resolve()
        abs_log_parent = Path(log_dir).parent.resolve()

        # Assertions
        assert abs_checkpoint_parent == abs_log_parent, (
            f"Parent of `checkpointing.checkpoint_dir` ({abs_checkpoint_parent}) and parent of `logger.log_dir` ({abs_log_parent}) must be the same folder."
        )
        assert abs_checkpoint_parent == abs_script_dir, (
            f"Parent of `checkpointing.checkpoint_dir`/`logger.log_dir` ({abs_checkpoint_parent}) must be the same folder as the script ({abs_script_dir})."
        )

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
