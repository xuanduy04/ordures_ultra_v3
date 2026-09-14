

# This module tests things outside of any package (e.g., things in the root __init__.py)

import os


def test_usage_stats_disabled_by_default():
    assert os.environ["RAY_USAGE_STATS_ENABLED"] == "0", (
        "Our dockerfile, slurm submission script and default environment setting when importing nemo rl should all disable usage stats collection. This failing is not expected."
    )


def test_usage_stats_disabled_in_tests():
    assert os.environ["RAY_USAGE_STATS_ENABLED"] == "0", (
        "Our dockerfile, slurm submission script and default environment setting when importing nemo rl should all disable usage stats collection. This failing is not expected."
    )
