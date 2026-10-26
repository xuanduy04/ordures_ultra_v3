"""Stdlib regression: run the renderer expression through a symlinked venv.

The temporary fixture contains only pyvenv.cfg and one marker module. No pip,
dependency installation, Ray or Gym server is used.
"""

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[3]


def rendered_python(path):
    tree = ast.parse((ROOT / "tools/xtoken_all/render.py").read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    expression = next(keyword.value for node in calls for keyword in node.keywords
                      if keyword.arg == "XTOKEN_GYM_ACTOR_PYTHON")
    return eval(compile(ast.Expression(body=expression), "renderer_python_path", "eval"),
                {"os": os, "str": str, "args": SimpleNamespace(gym_actor_python=path)})


class InterpreterPathTests(unittest.TestCase):
    def test_absolute_path_normalizes_without_dereferencing(self):
        relative = Path("environment/bin/python")
        self.assertEqual(rendered_python(relative), os.path.abspath(relative))

    @unittest.skipIf(os.name == "nt", "Linux symlink/pyvenv execution fixture")
    def test_symlink_entrypoint_keeps_venv_and_its_packages(self):
        with tempfile.TemporaryDirectory(prefix="xtoken-venv-path-") as temporary:
            root = Path(temporary)
            binary = root / "bin/python"
            binary.parent.mkdir()
            binary.symlink_to(Path(sys.executable).resolve())
            (root / "pyvenv.cfg").write_text(
                "home = " + str(Path(sys.executable).resolve().parent) + "\n"
                "include-system-site-packages = false\n")
            packages = root / ("lib/python" + str(sys.version_info.major) + "." + str(sys.version_info.minor)) / "site-packages"
            packages.mkdir(parents=True)
            (packages / "xtoken_venv_marker.py").write_text("VALUE = 'only-in-this-venv'\n")
            env = dict(os.environ)
            for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
                env.pop(key, None)
            command = "import json,sys,xtoken_venv_marker as m; print(json.dumps({'prefix':sys.prefix,'marker':m.VALUE}))"
            fixed = subprocess.run([rendered_python(binary), "-B", "-c", command],
                                   env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(fixed.returncode, 0, fixed.stderr)
            report = json.loads(fixed.stdout)
            self.assertEqual(Path(report["prefix"]).resolve(), root.resolve())
            self.assertEqual(report["marker"], "only-in-this-venv")
            original = subprocess.run([str(binary.resolve()), "-B", "-c", command],
                                      env=env, capture_output=True, text=True, timeout=15)
            self.assertNotEqual(original.returncode, 0)
            self.assertIn("ModuleNotFoundError", original.stderr)


if __name__ == "__main__":
    unittest.main()
