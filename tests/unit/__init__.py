


"""
Auto-loading remote_select plugin here:
- Ensures the plugin is discovered without extra CLI flags or global config.
- Loads early in pytest’s startup so ``pytest_load_initial_conftests`` can
  rewrite args before other plugins (e.g., testmon) prune collection.
- Scopes behavior to unit tests only (does not affect functional tests).
- Avoids a top-level ``conftest.py`` that would apply repo-wide.
"""

pytest_plugins = ["tests.unit._plugins.remote_select"]
