"""Probe the packages needed by each worker without installing their runtime."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("role_runtime_probe", ROOT / "tools/xtoken_all/runtime_probe.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


@pytest.fixture
def generation(monkeypatch):
    class Worker:
        def wake_up_async(self):
            pass

        def sleep_async(self):
            pass

        def _setup_vllm_openai_api_server(self):
            pass

    module = SimpleNamespace(
        __file__=ROOT / "nemo_rl/models/generation/vllm/vllm_worker_async.py",
        VllmAsyncGenerationWorker=Worker,
    )
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: module)
    return module


def test_generation_works_without_automodel_metadata_or_import(generation, monkeypatch):
    requested = []

    def version(name):
        requested.append(name)
        if name == "nemo-automodel":
            raise runtime.importlib.metadata.PackageNotFoundError(name)
        return "installed-version"

    monkeypatch.setattr(runtime.importlib.metadata, "version", version)
    result = runtime.probe("generation", None)
    assert requested == ["torch", "transformers", "ray", "vllm"]
    assert "nemo-automodel" not in result
    assert result["vllm"] == "installed-version"
    assert result["cuda_operations_requested"] is False


@pytest.mark.parametrize("missing", ["torch", "transformers", "ray", "vllm"])
def test_generation_still_requires_its_packages(generation, monkeypatch, missing):
    def version(name):
        if name == missing:
            raise runtime.importlib.metadata.PackageNotFoundError(name)
        return "installed-version"

    monkeypatch.setattr(runtime.importlib.metadata, "version", version)
    with pytest.raises(runtime.importlib.metadata.PackageNotFoundError):
        runtime.probe("generation", None)


@pytest.mark.parametrize("missing", ["wake_up_async", "sleep_async", "_setup_vllm_openai_api_server"])
def test_generation_still_requires_async_apis(generation, monkeypatch, missing):
    setattr(generation.VllmAsyncGenerationWorker, missing, None)
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda name: "installed-version")
    with pytest.raises(TypeError, match="Missing async generation API"):
        runtime.probe("generation", None)


def test_ray_decorated_generation_class_is_unwrapped(generation, monkeypatch):
    generation.VllmAsyncGenerationWorker = SimpleNamespace(
        __ray_metadata__=SimpleNamespace(modified_class=generation.VllmAsyncGenerationWorker))
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda name: "installed-version")
    assert runtime.probe("generation", None)["role"] == "generation"


def test_worker_source_must_still_match_checkout(generation, monkeypatch, tmp_path):
    generation.__file__ = tmp_path / "vllm_worker_async.py"
    with pytest.raises(ValueError, match="different NeMo source"):
        runtime.probe("generation", None)


def test_policy_still_requires_automodel_metadata(monkeypatch):
    module = SimpleNamespace(__file__=ROOT / "nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py")
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: module)
    requested = []

    def version(name):
        requested.append(name)
        if name == "nemo-automodel":
            raise runtime.importlib.metadata.PackageNotFoundError(name)
        return "installed-version"

    monkeypatch.setattr(runtime.importlib.metadata, "version", version)
    with pytest.raises(runtime.importlib.metadata.PackageNotFoundError):
        runtime.probe("policy", None)
    assert requested == ["torch", "transformers", "ray", "nemo-automodel"]


def test_actor_probe_imports_full_actor_and_sdk_without_generation_packages(monkeypatch, tmp_path):
    class Actor:
        def _spinup(self):
            pass

        async def run_rollouts(self):
            yield None

        def shutdown(self):
            pass

    sdk = SimpleNamespace(PARENT_DIR=tmp_path, __file__=tmp_path / "nemo_gym/__init__.py")
    actor_module = SimpleNamespace(__file__=ROOT / "nemo_rl/environments/nemo_gym.py",
        NemoGym=SimpleNamespace(__ray_metadata__=SimpleNamespace(modified_class=Actor)))
    modules = {"nemo_rl.environments.nemo_gym": actor_module, "nemo_gym": sdk,
        "nemo_rl.algorithms.x_token.gym_compat": SimpleNamespace(),
        "nemo_gym.cli": SimpleNamespace(RunHelper=object),
        "nemo_gym.rollout_collection": SimpleNamespace(RolloutCollectionHelper=object)}
    imported, versions = [], []

    def import_module(name):
        imported.append(name)
        return modules[name]

    def version(name):
        versions.append(name)
        if name in ("nemo-automodel", "vllm"):
            raise AssertionError("Actor must not require policy/generation-only packages")
        return "installed-version"

    monkeypatch.setattr(runtime.importlib, "import_module", import_module)
    monkeypatch.setattr(runtime.importlib.metadata, "version", version)
    report = runtime.probe("gym_actor", tmp_path)
    assert report["actor_import_completed"] is True
    assert report["checked_actor_apis"] == ["_spinup", "run_rollouts", "shutdown"]
    assert "nemo_rl.algorithms.x_token.gym_compat" in imported
    assert versions == ["torch", "transformers", "ray"]

    def synchronous_rollouts(self):
        return iter(())

    monkeypatch.setattr(Actor, "run_rollouts", synchronous_rollouts)
    with pytest.raises(TypeError, match="async generator"):
        runtime.probe("gym_actor", tmp_path)
