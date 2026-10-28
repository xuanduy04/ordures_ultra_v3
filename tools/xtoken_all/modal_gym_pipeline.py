"""Budget-bounded real-model Gym qualification on isolated Modal resources."""

from decimal import Decimal, ROUND_CEILING
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import secrets
import shutil
import signal
import time
import threading
import urllib.error
import urllib.request

import modal


ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/work/source")
CONFIG = json.loads(Path(os.environ["XTOKEN_GYM_CONFIG_JSON"]).read_text()) if modal.is_local() else None
IMAGE_DIGEST = "nvcr.io/nvidia/nemo-rl@sha256:060a3337d167e66230ce8b3dac9e9b733ad25bc997f8a0121eebccf81cbf1b23"
IMAGE = modal.Image.from_registry(IMAGE_DIGEST)
if modal.is_local():
    for relative in ("nemo_rl", "tools/xtoken_all", "examples", "custom_utils"):
        IMAGE = IMAGE.add_local_dir(str(ROOT / relative), "/work/source/" + relative, ignore=["**/__pycache__/**"])
    for name in ("pyproject.toml", "uv.lock", "default_runtime_env.yaml", ".python-version"):
        IMAGE = IMAGE.add_local_file(str(ROOT / name), "/work/source/" + name)
    overlay = Path(os.environ["XTOKEN_GYM_OVERLAY"])
    IMAGE = IMAGE.add_local_dir(str(overlay / "Gym"), "/work/gym", ignore=["**/__pycache__/**"])
    IMAGE = IMAGE.add_local_dir(str(overlay / "nltk_data"), "/work/nltk_data")
VOLUME = modal.Volume.from_name("xtoken-real-gym-20261028", create_if_missing=True)
app = modal.App("xtoken-real-gym-20261028")
PYTHON = "/opt/nemo_rl_venv/bin/python"
TEACHER_IMAGE = modal.Image.from_registry("vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b", add_python="3.12").entrypoint([])


def environment():
    env = {key: value for key, value in os.environ.items() if key not in {
        "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_IDENTITY_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"}}
    env.update(PYTHONPATH="/cache/supplement:/work/source:/work/gym", PYTHONDONTWRITEBYTECODE="1",
               NRL_IGNORE_VERSION_MISMATCH="1", HF_HOME="/cache/hf", HF_HUB_DISABLE_TELEMETRY="1",
               CUSTOM_NLTK_DATA_DIR="/work/nltk_data", NLTK_DATA="/work/nltk_data", OMP_NUM_THREADS="2",
               VLLM_USE_DEEP_GEMM="0", VLLM_MOE_USE_DEEP_GEMM="0")
    return env


def execute_worker(command, env, timeout, key):
    """Stream real phase/metric receipts while retaining sanitized full stdout."""
    process = subprocess.Popen(command, env=env, cwd="/work/source", stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, start_new_session=True)
    timed_out = threading.Event()

    def stop():
        timed_out.set()
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    timer = threading.Timer(timeout, stop)
    timer.start()
    lines = []
    try:
        for line in process.stdout:
            clean = line.replace(key, "[redacted]") if key else line
            if any(name in clean for name in ("env_vars", "MODAL_IDENTITY_TOKEN", "TOKEN_SECRET")):
                clean = "[environment line omitted]\n"
            lines.append(clean)
            if any(marker in clean for marker in ("GYM_", "XTOKEN_TRAIN_STEP=", "XTOKEN_TEACHER_SCORE=", "XTOKEN_ROLLOUT=",
                                                  "XTOKEN_REFIT_REQUEST=", "NRL_REFIT_BUFFER=", "XTK_TEACHER_", "PROBE=", "Error", "Traceback",
                                                  'Saving checkpoint for step', 'Training Results:', 'Loss:', 'Max number of steps')):
                print(clean, end="", flush=True)
        code = process.wait()
    finally:
        timer.cancel()
    return {"exit_code": 124 if timed_out.is_set() else code, "stdout": "".join(lines), "stderr": ""}


@app.function(image=IMAGE, volumes={"/cache": VOLUME}, cpu=(8, 8), memory=(32768, 32768),
              timeout=1200, startup_timeout=120, scaledown_window=2, retries=0)
def stage(arm, phase):
    command = [PYTHON, "-B", "/work/source/tools/xtoken_all/gym_real_worker.py", "--phase", phase, "--arm", arm]
    result = execute_worker(command, environment(), 1150, None)
    receipt = Path("/cache/imports-" + arm + ".json")
    result["receipt"] = json.loads(receipt.read_text()) if receipt.is_file() else None
    VOLUME.commit()
    if result["exit_code"]:
        print(result["stdout"][-14000:], flush=True)
    return result


@app.function(image=TEACHER_IMAGE, volumes={"/cache": VOLUME}, cpu=(1, 1), memory=(2048, 2048),
              timeout=60, startup_timeout=120, scaledown_window=2, retries=0)
def teacher_inventory():
    executable = shutil.which("vllm")
    if executable is None:
        candidates = [path for root in ("/opt", "/usr/local/bin", "/usr/bin") for path in Path(root).rglob("vllm")
                      if path.is_file() and path.parent.name == "bin"]
        if len(candidates) != 1:
            raise RuntimeError("Cannot identify the isolated teacher's existing vLLM CLI")
        executable = str(candidates[0])
    shebang = Path(executable).read_text().splitlines()[0]
    if not shebang.startswith("#!/") or " " in shebang:
        raise RuntimeError("Teacher CLI requires an explicit interpreter path")
    result = subprocess.run([shebang[2:], "-B", "-c", "import importlib.metadata as m,json,sys;print(json.dumps({'python':sys.executable,'vllm':m.version('vllm'),'torch':m.version('torch'),'transformers':m.version('transformers')}))"],
                            capture_output=True, text=True, check=True, timeout=30)
    report = {"cli": executable, **json.loads(result.stdout)}
    if report["vllm"] != "0.31.0":
        raise ValueError("Teacher image has another vLLM version")
    Path("/cache/teacher-runtime.json").write_text(json.dumps(report, indent=2))
    VOLUME.commit()
    return report


def metadata_gateway(key, listen, upstream):
    """Super QA reads model metadata without auth; all scoring stays authenticated.

    Forward actual server replies unchanged. This test-only gateway avoids
    changing the reviewed Gym source or publishing an unauthenticated scorer.
    """
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.forward_request(False)

        def do_POST(self):
            self.forward_request(True)

        def forward_request(self, authenticated):
            if self.path not in (("/tokenize", "/v1/completions") if authenticated else ("/version", "/v1/models")):
                self.send_error(404)
                return
            authorization = self.headers["Authorization"] if "Authorization" in self.headers else ""
            if authenticated and not hmac.compare_digest(authorization, "Bearer " + key):
                self.send_error(401)
                return
            count = int(self.headers["Content-Length"]) if "Content-Length" in self.headers else 0
            if count < 0 or count > 4 * 1024 * 1024:
                self.send_error(413)
                return
            body = self.rfile.read(count) if authenticated else None
            request = urllib.request.Request(upstream + self.path, data=body,
                                             headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
            try:
                response = urllib.request.urlopen(request, timeout=180)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                payload = response.read(67108865)
                if len(payload) > 67108864:
                    self.send_error(502)
                    return
                self.send_response(response.code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

    server = ThreadingHTTPServer(listen, Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def legacy_loopback_gateway(key, upstream):
    """Forward the unchanged old client through an authenticated owned teacher.

    Only the learner container's loopback interface accepts unauthenticated
    legacy requests. The public teacher scorer still requires its ephemeral key.
    """
    receipts = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == '/_receipts':
                body = json.dumps(receipts).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path in ('/v1/models', '/version'):
                self.forward(None)
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != '/v1/completions':
                self.send_error(404)
                return
            count = int(self.headers['Content-Length'])
            if count < 0 or count > 4 * 1024 * 1024:
                self.send_error(413)
                return
            body = self.rfile.read(count)
            value = json.loads(body)
            if value['max_tokens'] != 0 or value['echo'] is not True:
                self.send_error(400)
                return
            self.forward(body)

        def forward(self, body):
            request = urllib.request.Request(upstream + self.path, data=body,
                headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
            try:
                response = urllib.request.urlopen(request, timeout=180)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                payload = response.read(67108865)
                if len(payload) > 67108864:
                    self.send_error(502)
                    return
                if body is not None:
                    receipts.append({'status': response.code, 'response_bytes': len(payload),
                                     'scoring_only': True})
                self.send_response(response.code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@app.function(image=TEACHER_IMAGE, volumes={"/cache": VOLUME}, cpu=(16, 16), memory=(131072, 131072),
              timeout=2100, startup_timeout=180, scaledown_window=2, retries=0)
def teacher_server(arm, key, ready, control):
    """Ephemeral authenticated teacher; neither deploys nor contacts company nodes."""
    ready.put({"state": "allocated"})
    VOLUME.reload()
    spec = json.loads(Path("/cache/stage-" + arm + ".json").read_text())["models"]["teacher"]
    alias = spec["alias"]
    executable = json.loads(Path("/cache/teacher-runtime.json").read_text())["cli"]
    command = [executable, "serve", spec["path"], "--served-model-name", alias, "--host", "0.0.0.0", "--port", "8000",
               "--dtype", "bfloat16", "--tensor-parallel-size", "1", "--max-model-len", "512", "--max-num-seqs", "4",
               "--max-num-batched-tokens", "1024", "--gpu-memory-utilization", "0.94", "--enforce-eager",
               "--max-logprobs", "20", "--logprobs-mode", "raw_logprobs"]
    if arm == "cross":
        command.append("--language-model-only")
    env = dict(os.environ, VLLM_API_KEY=key, HF_HUB_OFFLINE="1", OMP_NUM_THREADS="2")
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_IDENTITY_TOKEN",
                 "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        env.pop(name, None)
    log_path = Path("/tmp/teacher.log")
    with log_path.open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 330
            while True:
                try:
                    request = urllib.request.Request("http://127.0.0.1:8000/v1/models", headers={"Authorization": "Bearer " + key})
                    with urllib.request.urlopen(request, timeout=4) as response:
                        models = json.load(response)
                    if alias not in {item["id"] for item in models["data"]}:
                        raise RuntimeError("Teacher served another model alias")
                    request = urllib.request.Request("http://127.0.0.1:8000/version", headers={"Authorization": "Bearer " + key})
                    with urllib.request.urlopen(request, timeout=4) as response:
                        version = json.load(response)["version"]
                    if version != "0.31.0":
                        raise RuntimeError("Pinned test teacher unexpectedly reports version " + version)
                    break
                except (urllib.error.URLError, TimeoutError):
                    if process.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError("Teacher failed startup: " + log_path.read_text(errors="replace")[-6000:].replace(key, "[redacted]"))
                    time.sleep(2)
            gateway = metadata_gateway(key, ("0.0.0.0", 8001), "http://127.0.0.1:8000")
            try:
                with modal.forward(8001) as tunnel:
                    ready.put({"state": "ready", "url": tunnel.url, "alias": alias, "logprobs_mode": "raw_logprobs"})
                    print("GYM_TEACHER_READY=" + alias, flush=True)
                    control.get(timeout=1740)
            finally:
                gateway.shutdown()
                gateway.server_close()
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                pass
    return {"arm": arm, "log": log_path.read_text(errors="replace").replace(key, "[redacted]"), "command": command}


@app.function(image=IMAGE, volumes={"/cache": VOLUME}, cpu=(16, 16), memory=(131072, 131072),
              timeout=1500, startup_timeout=180, scaledown_window=2, retries=0)
def student_pipeline(arm, url, key, ready, selected, execution_limit):
    ready.put({"state": "student_allocated"})
    VOLUME.reload()
    summary_path = Path('/cache/evidence') / arm / 'summary.json'
    summary_path.unlink(missing_ok=True)
    VOLUME.commit()
    env = environment()
    env.update(XTOKEN_TEACHER_URL=url, XTOKEN_TEACHER_API_KEY=key)
    relay = legacy_loopback_gateway(key, url)
    env['XTOKEN_LEGACY_TEACHER_URL'] = 'http://127.0.0.1:' + str(relay.server_port)
    env.update(XTOKEN_GYM_SELECTED_MODES=json.dumps(selected),
               XTOKEN_GYM_SUITE_SECONDS=str(execution_limit - (40 if len(selected) == 1 else 130)))
    command = [PYTHON, "-B", "/work/source/tools/xtoken_all/gym_real_worker.py", "--phase", "canary", "--arm", arm]
    try:
        result = execute_worker(command, env, execution_limit - (20 if len(selected) == 1 else 70), key)
    finally:
        relay.shutdown()
        relay.server_close()
        root = Path("/tmp/xtoken-real") / arm
        if root.is_dir():
            for path in root.rglob("*"):
                if path.is_file() and "checkpoints" not in path.parts and path.suffix in (".log", ".yaml", ".json"):
                    destination = Path("/cache/evidence") / arm / path.relative_to(root)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    content = path.read_text(errors="replace").replace(key, "[redacted]")
                    content = "\n".join(line if "env_vars" not in line else "[environment line omitted]" for line in content.splitlines())
                    destination.write_text(content)
        VOLUME.commit()
    print(result["stdout"][-22000:], flush=True)
    print(result["stderr"][-10000:], flush=True)
    summary = Path("/cache/evidence") / arm / "summary.json"
    result["summary"] = json.loads(summary.read_text()) if summary.is_file() else None
    return result


@contextmanager
def locked(ledger):
    lock = ledger.with_suffix(".lock")
    deadline = time.monotonic() + 10
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            if time.monotonic() > deadline:
                raise RuntimeError("Budget ledger already locked; fail closed")
            time.sleep(.1)
    try:
        yield
    finally:
        lock.rmdir()


def cumulative_upper(reported, baseline, reservations):
    """Do not add already billed terminal reservations a second time."""
    return max(Decimal(reported), Decimal(baseline) + sum((Decimal(row["upper_bound_usd"]) for row in reservations), Decimal(0)))


def cpu_elapsed_bound(seconds):
    elapsed = Decimal(str(seconds))
    if not elapsed.is_finite() or elapsed < 0:
        raise ValueError("Invalid completed CPU duration")
    rate = 8 * Decimal("0.0000131") + 32 * Decimal("0.00000222")
    return ((elapsed + 30) * rate + Decimal("0.03")).quantize(Decimal("0.01"), rounding=ROUND_CEILING)


def gpu_role_bound(accelerator, count, teacher_s, student_s, overhead_s, margin):
    if accelerator not in ('H100', 'B200') or count not in (2, 3, 4):
        raise ValueError('Unknown priced GPU topology')
    if min(teacher_s, student_s, overhead_s) < 0 or teacher_s + student_s <= 0:
        raise ValueError('GPU lifetimes must be positive for at least one reserved role')
    rate = Decimal('0.001097') if accelerator == 'H100' else Decimal('0.001736')
    host = 16 * Decimal('0.0000131') + 128 * Decimal('0.00000222')
    bound = Decimal(str(margin))
    if teacher_s:
        bound += (teacher_s + overhead_s) * (rate + host)
    if student_s:
        bound += (student_s + overhead_s) * (count * rate + host)
    return bound.quantize(Decimal('0.1'), rounding=ROUND_CEILING)


def settle_cpu_history(ledger, output, profile):
    """Tighten completed CPU reservations from retained duration evidence only."""
    if not ledger.is_file():
        return
    with locked(ledger):
        records = json.loads(ledger.read_text())
        for row in records:
            if row["phase"] not in ('stage', 'imports') or "id" not in row:
                continue
            receipt = output / (profile + "-" + row["arm"] + "-" + row["phase"] + "-" + row["id"] + ".json")
            if not receipt.is_file():
                continue
            raw = receipt.read_bytes()
            result = json.loads(raw)
            if result["profile"] != profile or result["arm"] != row["arm"] or "exit_code" not in result:
                raise ValueError("CPU settlement receipt differs from the reserved call")
            bound = min(Decimal(row["upper_bound_usd"]), cpu_elapsed_bound(result["elapsed_seconds"]))
            if "initial_upper_bound_usd" not in row:
                row["initial_upper_bound_usd"] = row["upper_bound_usd"]
            row.update(upper_bound_usd=str(bound), elapsed_seconds=result["elapsed_seconds"],
                       settlement="completed CPU8/32GiB submit-to-result plus30s; never billed actual",
                       settlement_receipt_sha256=hashlib.sha256(raw).hexdigest())
        ledger.write_text(json.dumps(records, indent=2))


def run_arm(arm, accelerator, selected, teacher_limit, student_limit):
    key = secrets.token_urlsafe(32)
    with modal.Queue.ephemeral() as ready, modal.Queue.ephemeral() as control:
        teacher = teacher_server.with_options(gpu=accelerator, timeout=teacher_limit).spawn(arm, key, ready, control)
        student = None
        teacher_result = None
        result = {"exit_code": 1, "summary": None}
        teacher_seconds = time.monotonic()
        student_seconds = None
        allocated = False

        def receipt(timeout):
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Teacher receipt deadline exceeded")
                try:
                    return ready.get(timeout=min(15, remaining))
                except queue.Empty:
                    try:
                        teacher.get(timeout=0)
                    except TimeoutError:
                        continue
                    raise RuntimeError("Teacher exited before its readiness receipt")

        try:
            allocation = receipt(120)
            if allocation["state"] != "allocated":
                raise RuntimeError("Missing teacher allocation receipt")
            allocated = True
            print("GYM_ALLOCATION=" + json.dumps({"role": "teacher", "accelerator": accelerator}), flush=True)
            connection = receipt(360)
            if connection["state"] != "ready":
                raise RuntimeError("Missing teacher readiness receipt")
            student_seconds = time.monotonic()
            count = CONFIG["canary"]["gpu_count"]
            student = student_pipeline.with_options(gpu=accelerator + ":" + str(count), timeout=student_limit).spawn(
                arm, connection["url"], key, ready, selected, student_limit)
            student_allocation = ready.get(timeout=120)
            if student_allocation["state"] != "student_allocated":
                raise RuntimeError("Missing student allocation receipt")
            print("GYM_ALLOCATION=" + json.dumps({"role": "student", "accelerator": accelerator, "count": count}), flush=True)
            result = student.get(timeout=student_limit + 200)
        except Exception as error:
            result = {"exit_code": 1, "summary": None, "error_type": type(error).__name__,
                      "error": str(error).replace(key, "[redacted]")[-12000:]}
        finally:
            if student is not None:
                student.cancel(terminate_containers=True)
            control.put("stop")
            try:
                teacher_result = teacher.get(timeout=20)
            except Exception as error:
                teacher_result = {"error_type": type(error).__name__, "error": str(error).replace(key, "[redacted]")[-12000:]}
                teacher.cancel(terminate_containers=True)
        now = time.monotonic()
        return {"student": result, "teacher": teacher_result, "accelerator_requested": accelerator,
                "teacher_allocated_receipt": allocated, "teacher_submit_to_stop_s": now - teacher_seconds,
                "student_submit_to_stop_s": now - student_seconds if student_seconds is not None else 0}


def run_teacher_session(arm, accelerator, limit, session_path, guard):
    """Reserve teacher GPU on one profile while a separately reserved learner uses it."""
    key = secrets.token_urlsafe(32)
    path = Path(session_path)
    if not path.is_absolute() or not str(path).startswith('/tmp/') or path.exists():
        raise ValueError('Use a fresh private /tmp session file, never a repository/artifact path')
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    started = time.monotonic()
    with modal.Queue.ephemeral() as ready, modal.Queue.ephemeral() as control:
        teacher = teacher_server.with_options(gpu=accelerator, timeout=limit).spawn(arm, key, ready, control)
        try:
            if ready.get(timeout=120)['state'] != 'allocated':
                raise RuntimeError('Missing teacher allocation receipt')
            connection = ready.get(timeout=360)
            if connection['state'] != 'ready':
                raise RuntimeError('Teacher did not become ready')
            descriptor = {'url': connection['url'], 'key': key, 'arm': arm, 'profile': os.environ['MODAL_PROFILE'],
                          'app_id': app.app_id, 'created_at': time.time(), 'guard': guard}
            pending = path.with_suffix('.pending')
            descriptor_fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor_fd, 'w') as stream:
                json.dump(descriptor, stream)
            pending.replace(path)
            print('GYM_SHARED_TEACHER_READY=' + json.dumps({'arm': arm, 'app_id': app.app_id}), flush=True)
            while time.monotonic() - started < limit - 20 and not path.with_suffix('.stop').exists():
                time.sleep(1)
        finally:
            control.put('stop')
            try:
                teacher.get(timeout=20)
            except Exception:
                teacher.cancel(terminate_containers=True)
            path.unlink(missing_ok=True)
            path.with_suffix('.pending').unlink(missing_ok=True)
    return {'exit_code': 0, 'teacher_submit_to_stop_s': time.monotonic() - started,
            'student_submit_to_stop_s': 0, 'teacher_session': True}


def run_shared_learner(arm, accelerator, selected, limit, session_path):
    path = Path(session_path)
    descriptor = json.loads(path.read_text())
    if descriptor['arm'] != arm or time.time() - descriptor['created_at'] > 600 or not descriptor['guard']['allowed']:
        raise ValueError('Missing current separately reserved teacher session')
    if descriptor['profile'] == os.environ['MODAL_PROFILE']:
        raise ValueError('Shared session requires explicit separate profile accounting')
    started = time.monotonic()
    with modal.Queue.ephemeral() as ready:
        student = None
        try:
            student = student_pipeline.with_options(gpu=accelerator + ':' + str(CONFIG['canary']['gpu_count']),
                                                   timeout=limit).spawn(
                arm, descriptor['url'], descriptor['key'], ready, selected, limit)
            if ready.get(timeout=120)['state'] != 'student_allocated':
                raise RuntimeError('Missing learner allocation receipt')
            result = student.get(timeout=limit + 200)
        finally:
            if student is not None:
                student.cancel(terminate_containers=True)
            path.with_suffix('.stop').touch()
    return {'student': result, 'teacher_profile': descriptor['profile'], 'teacher_app_id': descriptor['app_id'],
            'teacher_submit_to_stop_s': 0, 'student_submit_to_stop_s': time.monotonic() - started}


@app.local_entrypoint()
def main(phase: str, arm: str, output_dir: str, accelerator: str | None = None, only_mode: str | None = None,
         mode_set: str | None = None, teacher_session: str | None = None):
    if accelerator is None:
        accelerator = CONFIG["accelerator"]
    profile = os.environ["MODAL_PROFILE"]
    if profile not in ("nguyenngocminh30012005", "nguyenepti") or arm not in ("cross", "same"):
        raise ValueError("Use an authorized profile and explicit cross/same arm")
    if phase not in ("stage", "imports", "gpu", "teacher", "learner") or accelerator not in ("B200", "H100"):
        raise ValueError("Unknown phase")
    supported = [*CONFIG['modes'], CONFIG['legacy_mode']]
    if only_mode is not None and (phase not in ('gpu', 'learner') or only_mode not in supported):
        raise ValueError("Single-mode retry requires an explicit configured GPU mode")
    selected = CONFIG["modes"] if only_mode is None else [only_mode]
    if mode_set is not None:
        selected = mode_set.split(',')
        if only_mode is not None or not selected or len(set(selected)) != len(selected) or not set(selected).issubset(supported):
            raise ValueError('Invalid explicit mode set')
    if phase in ('teacher', 'learner') and teacher_session is None:
        raise ValueError('A private teacher session is required for separately billed roles')
    limits = CONFIG["qualification_limits"]
    teacher_limit = limits["teacher_s"] if only_mode is None else limits["single_teacher_s"]
    student_limit = limits["student_s"] if only_mode is None else limits["single_student_s"]
    if phase in ('teacher', 'learner'):
        teacher_limit, student_limit = limits['shared_teacher_s'], limits['shared_student_s']
    count = CONFIG["canary"]["gpu_count"]
    if count not in (2, 3, 4):
        raise ValueError("Use an explicit two-, three-, or four-GPU student topology")
    if CONFIG["canary"]["global_batch_size"] % count or CONFIG["canary"]["max_num_steps"] != 2:
        raise ValueError("This two-update canary requires a global batch divisible by the physical GPU count")
    if Decimal(str(CONFIG["budget_usd"])) > Decimal("20") or Decimal(str(CONFIG["billing_lag_reserve_usd"])) < Decimal("2"):
        raise ValueError("YAML cannot relax the user's USD20 profile ceiling or its USD2 lag reserve")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if phase in ('gpu', 'teacher', 'learner'):
        apps = json.loads(subprocess.check_output(["modal", "app", "list", "--json"], text=True))
        if any(row["app_id"] != app.app_id and int(row["tasks"]) > 0 for row in apps):
            raise RuntimeError("Another active app has unbounded overlapping spend; wait for it or account for its reservation")
        candidates = [output / (profile + "-" + arm + "-" + name + ".json") for name in ("stage", "imports")]
        existing = [path for path in candidates if path.is_file()]
        prepared = json.loads(max(existing, key=lambda path: path.stat().st_mtime).read_text()) if existing else None
        if prepared is None or prepared["exit_code"] != 0 or "receipt" not in prepared or prepared["receipt"] is None:
            raise RuntimeError("A successful current CPU preparation receipt is required before GPU allocation")
        receipt = prepared["receipt"]
        if receipt["success"] is not True or receipt["supplement_packages"] != CONFIG["supplement"]["packages"]:
            raise RuntimeError("CPU preparation is stale for the selected runtime supplement")
        if CONFIG['legacy_mode'] in selected and (not receipt['legacy_entrypoint_imported'] or not receipt['legacy_render_checked']):
            raise RuntimeError('The legacy entrypoint and actual rendered OPD loss must pass CPU preflight first')
    rows = json.loads(subprocess.check_output(["modal", "billing", "report", "--for", "this month", "--show-resources", "--json"], text=True))
    reported = sum((Decimal(row["cost"]) for row in rows), Decimal(0))
    # This preparation is CPU8/32GiB, <=1200s plus startup/teardown. GPU reservations
    # are separate and must be made together before either role starts.
    rate = Decimal("0.001736") if accelerator == "B200" else Decimal("0.001097")
    host_rate = 16 * Decimal("0.0000131") + 128 * Decimal("0.00000222")
    overhead = limits["startup_s"] + limits["teardown_s"]
    gpu_bound = gpu_role_bound(accelerator, count, teacher_limit, student_limit, overhead, '0.5')
    bound = gpu_bound if phase == "gpu" else Decimal("0.35")
    if phase == 'teacher':
        bound = gpu_role_bound(accelerator, count, teacher_limit, 0, overhead, '0.1')
    if phase == 'learner':
        bound = gpu_role_bound(accelerator, count, 0, student_limit, overhead, '0.1')
    ledger = output / (profile + "-cpu-reservations.json")
    settle_cpu_history(ledger, output, profile)
    prior = json.loads(ledger.read_text()) if ledger.is_file() else []
    usage_upper = cumulative_upper(reported, CONFIG["billing_baseline_usd"][profile], prior)
    guard = {"reported_month_usage_usd": str(reported), "cumulative_usage_upper_usd": str(usage_upper), "next_call_bound_usd": str(bound),
             "hard_limit_usd": str(CONFIG["budget_usd"]), "lag_reserve_usd": str(CONFIG["billing_lag_reserve_usd"])}
    allowed = usage_upper + bound + Decimal(str(CONFIG["billing_lag_reserve_usd"])) <= Decimal(str(CONFIG["budget_usd"]))
    guard["allowed"] = allowed
    print("GYM_BUDGET=" + json.dumps(guard), flush=True)
    if not allowed:
        raise RuntimeError("Cumulative profile budget guard rejected preparation")
    label = str(time.time_ns())
    with locked(ledger):
        current = json.loads(ledger.read_text()) if ledger.is_file() else []
        current_upper = cumulative_upper(reported, CONFIG["billing_baseline_usd"][profile], current)
        if current_upper + bound + Decimal(str(CONFIG["billing_lag_reserve_usd"])) > Decimal(str(CONFIG["budget_usd"])):
            raise RuntimeError("Concurrent reservation exhausted the cumulative profile budget")
        current.append({"id": label, "phase": phase, "arm": arm, "upper_bound_usd": str(bound), "app_id": app.app_id})
        ledger.write_text(json.dumps(current, indent=2))
    (output / (profile + "-" + phase + "-billing.json")).write_text(json.dumps(rows, indent=2))
    started = time.monotonic()
    if phase == 'teacher':
        result = run_teacher_session(arm, accelerator, teacher_limit, teacher_session, guard)
    elif phase == 'learner':
        result = run_shared_learner(arm, accelerator, selected, student_limit, teacher_session)
    else:
        result = run_arm(arm, accelerator, selected, teacher_limit, student_limit) if phase == "gpu" else stage.remote(arm, phase)
    if phase == "imports" and result["exit_code"] == 0:
        result["teacher_runtime"] = teacher_inventory.remote()
    result.update(profile=profile, arm=arm, elapsed_seconds=time.monotonic() - started, guard=guard,
                  app_id=app.app_id, selected_modes=selected, teacher_limit_s=teacher_limit, student_limit_s=student_limit)
    (output / (profile + "-" + arm + "-" + phase + "-" + label + ".json")).write_text(json.dumps(result, indent=2))
    (output / (profile + "-" + arm + "-" + phase + ".json")).write_text(json.dumps(result, indent=2))
    if phase in ('gpu', 'teacher', 'learner'):
        gpu_rate = Decimal("0.001736") if accelerator == "B200" else Decimal("0.001097")
        cpu_memory_rate = 16 * Decimal("0.0000131") + 128 * Decimal("0.00000222")
        ceiling = Decimal('0')
        if result['teacher_submit_to_stop_s']:
            ceiling += Decimal(str(result["teacher_submit_to_stop_s"] + 60)) * (gpu_rate + cpu_memory_rate)
        if result["student_submit_to_stop_s"]:
            ceiling += Decimal(str(result["student_submit_to_stop_s"] + 60)) * (count * gpu_rate + cpu_memory_rate)
        ceiling += Decimal("0.1")
        with locked(ledger):
            current = json.loads(ledger.read_text())
            row = next(item for item in current if "id" in item and item["id"] == label)
            row.update(upper_bound_usd=str(min(bound, ceiling)), elapsed_seconds=str(result["elapsed_seconds"]),
                       settlement="submit-to-result plus60s; conservative upper estimate, not billed actual")
            ledger.write_text(json.dumps(current, indent=2))
        success = result['exit_code'] == 0 if phase == 'teacher' else result["student"]["exit_code"] == 0 and result["student"]["summary"] is not None and result["student"]["summary"]["success"]
    else:
        success = result["exit_code"] == 0
    if not success:
        raise RuntimeError("Remote preparation failed; inspect persisted result before retry")
