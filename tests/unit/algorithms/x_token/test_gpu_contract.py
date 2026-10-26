"""Real CUDA loss/gradient and cross-process IPC checks; no downloaded models."""

import multiprocessing
import gc
from contextlib import contextmanager, nullcontext
import os
from pathlib import Path
import subprocess

import pytest
import torch

from nemo_rl.algorithms.x_token.ipc import ensure_teacher_ipc_buffer
from nemo_rl.algorithms.x_token.loss import CrossTokenizerDistillationLossFn
from nemo_rl.algorithms.x_token.loss_utils import rebuild_teacher_full_logits_from_ipc
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def handles(storage, payload):
    batch, sequence, vocab = storage.shape[1:]
    return [{"teacher_shards": [{"payload_ipc": payload, "buf_idx": 0,
              "sample_index_in_buf": index, "actual_shape": (sequence, vocab),
              "full_seq_len": sequence, "full_vocab_size": vocab,
              "vocab_start_index": 0, "vocab_end_index": vocab,
              "global_seq_start": 0}]} for index in range(batch)]


def _logit_producer(connection, teacher_cpu):
    torch.cuda.set_device(0)
    shape = teacher_cpu.shape
    storage, payload = ensure_teacher_ipc_buffer(None, None, 1, *shape,
                                                 torch.float32, torch.device("cuda:0"))
    storage[0].copy_(teacher_cpu)
    torch.cuda.synchronize()
    connection.send(handles(storage, payload))
    assert connection.recv() == "consumed"


@contextmanager
def exported_teacher(teacher):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_logit_producer, args=(child, teacher.cpu()))
    process.start()
    try:
        assert parent.poll(90), "IPC producer did not respond"
        yield parent.recv()
        gc.collect()
        torch.cuda.synchronize()
        parent.send("consumed")
        process.join(30)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("transport", ["ipc", "dense"])
def test_real_ipc_loss_and_gradient_match_dense_masked_kl(reverse, transport):
    torch.cuda.set_device(0)
    torch.manual_seed(19)
    batch_size, sequence, vocab = 2, 6, 12
    student = torch.randn(batch_size, sequence, vocab, device="cuda", requires_grad=True)
    teacher = torch.randn_like(student)
    mask = torch.tensor([[0, 0, 1, 1, 0, 0], [0, 1, 1, 1, 0, 0]], device="cuda")
    sample_mask = torch.tensor([1., 0.], device="cuda")
    data = BatchedDataDict(input_ids=torch.randint(vocab, (batch_size, sequence), device="cuda"),
                          input_lengths=torch.tensor([4, 4], device="cuda"), token_mask=mask,
                          sample_mask=sample_mask)
    config = {"gold_loss": False, "xtoken_loss": False, "temperature": 1.7,
              "vocab_topk": vocab, "uncommon_topk": vocab, "reverse_kl": reverse,
              "exact_token_match_only": True, "kl_loss_weight": 1., "ce_loss_scale": 0.,
              "dynamic_loss_scaling": False, "kd_loss_mode": "sum", "sum_weights_metric": None,
              "alpha": 1., "normalize_teacher_by_vocab": False, "student_vocab_size": vocab,
              "teacher_vocab_sizes": [vocab], "projection_matrix_paths": [None],
              "teacher_weights": [1.], "teacher_gold_loss": [None], "teacher_xtoken_loss": [None]}
    fn = CrossTokenizerDistillationLossFn(config)
    valid = (mask[:, 1:] * sample_mask[:, None]).sum()
    context = exported_teacher(teacher) if transport == "ipc" else nullcontext(teacher.log_softmax(-1).cpu())
    with context as entries:
        data["teacher_0_full_logits_ipc" if transport == "ipc" else "teacher_0_full_logits"] = entries
        loss, _ = fn(student, data, sample_mask.sum(), valid)
        loss.backward()
        actual = loss.detach().clone()
        del loss
    ref = student.detach().clone().requires_grad_(True)
    student_lp = torch.log_softmax(ref[:, :-1] / 1.7, -1)
    teacher_lp = torch.log_softmax(teacher[:, :-1] / 1.7, -1)
    if reverse:
        per_token = (student_lp.exp() * (student_lp - teacher_lp)).sum(-1)
    else:
        per_token = (teacher_lp.exp() * (teacher_lp - student_lp)).sum(-1)
    expected = (per_token * mask[:, 1:] * sample_mask[:, None]).sum() / valid * 1.7 ** 2
    expected.backward()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(student.grad, ref.grad, rtol=2e-5, atol=2e-6)
    assert student.grad[1].abs().sum() == 0
    assert student.grad[0, 0].abs().sum() == 0
    assert student.grad[0, 1].abs().sum() > 0


@pytest.mark.parametrize("transport", ["ipc", "dense"])
def test_cross_tokenizer_response_loss_gradient_matches_dense_projection(tmp_path, transport):
    torch.cuda.set_device(0)
    torch.manual_seed(83)
    student = torch.randn(2, 6, 8, device="cuda", requires_grad=True)
    teacher = torch.randn(2, 7, 10, device="cuda")
    indices = torch.stack([torch.arange(8), (torch.arange(8) + 3) % 10], -1)
    weights = torch.tensor([[0.7, 0.3]]).expand(8, 2).contiguous()
    projection_path = str(tmp_path / "projection.pt")
    torch.save({"indices": indices, "likelihoods": weights}, projection_path)
    config = {"gold_loss": False, "xtoken_loss": False, "temperature": 1.3,
              "vocab_topk": 10, "uncommon_topk": 10, "reverse_kl": False,
              "exact_token_match_only": True, "kl_loss_weight": 1., "ce_loss_scale": 0.,
              "dynamic_loss_scaling": False, "kd_loss_mode": "sum", "sum_weights_metric": None,
              "alpha": 1., "normalize_teacher_by_vocab": False, "student_vocab_size": 8,
              "teacher_vocab_sizes": [10], "projection_matrix_paths": [projection_path],
              "teacher_weights": [1.], "teacher_gold_loss": [None], "teacher_xtoken_loss": [None]}
    data = BatchedDataDict(input_ids=torch.randint(8, (2, 6), device="cuda"),
                          input_lengths=torch.tensor([5, 5], device="cuda"),
                          token_mask=torch.tensor([[0, 0, 1, 1, 1, 0]] * 2, device="cuda"),
                          sample_mask=torch.tensor([1., 0.], device="cuda"),
                          alignment_0_student_chunk_id=torch.tensor([[-1, -1, 0, 1, 2, -1]] * 2, device="cuda"),
                          alignment_0_teacher_chunk_id=torch.tensor([[-1, -1, -1, 0, 1, 2, -1]] * 2, device="cuda"),
                          alignment_0_pair_valid=torch.ones(2, 3, dtype=torch.bool, device="cuda"),
                          alignment_0_pair_is_correct=torch.ones(2, 3, dtype=torch.bool, device="cuda"))
    context = exported_teacher(teacher) if transport == "ipc" else nullcontext(teacher.log_softmax(-1).cpu())
    with context as entries:
        data["teacher_0_full_logits_ipc" if transport == "ipc" else "teacher_0_full_logits"] = entries
        loss, _ = CrossTokenizerDistillationLossFn(config)(student, data,
                  torch.tensor(1., device="cuda"), torch.tensor(3., device="cuda"))
        loss.backward()
        actual = loss.detach().clone()
        del loss
    ref = student.detach().clone().requires_grad_(True)
    matrix = torch.zeros(8, 10, device="cuda")
    matrix.scatter_add_(1, indices.cuda(), weights.cuda())
    projected = torch.softmax(ref[0, 1:4] / 1.3, -1) @ matrix
    q = projected.log()
    q = q - q.logsumexp(-1, keepdim=True)
    p = torch.log_softmax(teacher[0, 2:5] / 1.3, -1)
    expected = (p.exp() * (p - q)).sum(-1).mean() * 1.3 ** 2
    expected.backward()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(student.grad, ref.grad, rtol=2e-5, atol=2e-6)


def _producer(connection):
    torch.cuda.set_device(0)
    storage, payload = ensure_teacher_ipc_buffer(None, None, 1, 2, 4, 8,
                                                 torch.float32, torch.device("cuda:0"))
    for version in range(2):
        storage.fill_(version + 1)
        torch.cuda.synchronize()
        connection.send(handles(storage, payload))
        assert connection.recv() == "consumed"
        reused, reused_payload = ensure_teacher_ipc_buffer(storage, payload, 1, 2, 4, 8,
                                                            torch.float32, storage.device)
        assert reused.data_ptr() == storage.data_ptr() and reused_payload is payload
    connection.close()


def test_two_gpu_ipc_reads_latest_persistent_buffer_across_processes():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    torch.cuda.set_device(1)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    producer = context.Process(target=_producer, args=(child,))
    producer.start()
    try:
        for version in range(2):
            assert parent.poll(90), "Producer did not publish IPC handles"
            entries = parent.recv()
            rebuilt = rebuild_teacher_full_logits_from_ipc(entries, cp_group=None, device=1)
            torch.testing.assert_close(rebuilt, torch.full((2, 4, 8), float(version + 1), device="cuda:1"))
            del rebuilt
            gc.collect()
            torch.cuda.synchronize()
            parent.send("consumed")
        producer.join(30)
        assert producer.exitcode == 0
    finally:
        if producer.is_alive():
            producer.terminate()
            producer.join(10)
        parent.close()
        torch.cuda.set_device(0)


def test_native_dtensor_worker_export_and_optimizer_update(tmp_path):
    executable = Path("/opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python")
    if not executable.is_file():
        pytest.skip("Native production worker interpreter not present in this image")
    world = min(torch.cuda.device_count(), 2)
    script = Path(__file__).with_name("native_worker_smoke.py")
    command = [str(executable), "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={world}", str(script)]
    environment = {**os.environ, "XTOKEN_FIXTURE_PATH": str(tmp_path / "tiny-llama"),
                   "HF_HUB_OFFLINE": "1", "UV_OFFLINE": "1", "OMP_NUM_THREADS": "1"}
    result = subprocess.run(command, env=environment, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=240)
    print(result.stdout, flush=True)
    assert result.returncode == 0, result.stdout
    assert result.stdout.count("XTOKEN_NATIVE_WORKER=") == world
