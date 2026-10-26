"""Independent coarse-distribution references, gradients and capped HTTP API."""

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest
import torch

from nemo_rl.algorithms.x_token.remote_teacher import XTokenTopKLogprobClient, TeacherCapabilityError
from nemo_rl.algorithms.x_token.topk_tail import ProjectedTopKTailLoss, coarse_kl
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def config(path):
    return {"temperature": 1., "gold_loss": False, "xtoken_loss": False,
            "kd_loss_mode": "sum", "dynamic_loss_scaling": False, "ce_loss_scale": 0.,
            "sum_weights_metric": None, "normalize_teacher_by_vocab": False,
            "teacher_gold_loss": [None], "teacher_xtoken_loss": [None],
            "projection_matrix_paths": [path], "student_vocab_size": 3, "teacher_vocab_sizes": [4],
            "teacher_weights": [1.], "kl_loss_weight": 1., "tail_epsilon": 1e-8,
            "reverse_kl": False, "exact_token_match_only": True}


def reference(student, teacher, selected, reverse):
    p = torch.cat([teacher[selected], (1 - teacher[selected].sum()).reshape(1)])
    q = torch.cat([student[selected], (1 - student[selected].sum()).reshape(1)])
    return (q * (q.log() - p.log())).sum() if reverse else (p * (p.log() - q.log())).sum()


@pytest.mark.parametrize("reverse", [False, True])
def test_coarsened_kl_matches_reference_and_is_bounded_by_dense(reverse):
    logits = torch.tensor([.7, -.2, .4, -1.], dtype=torch.float64, requires_grad=True)
    p = torch.tensor([.1, .2, .3, .4], dtype=torch.float64)
    selected = torch.tensor([0, 2])
    q = logits.softmax(-1)
    result = coarse_kl(q[selected][None], p[selected][None], reverse=reverse, epsilon=1e-12).sum()
    expected = reference(q, p, selected, reverse)
    dense = (q * (q.log() - p.log())).sum() if reverse else (p * (p.log() - q.log())).sum()
    torch.testing.assert_close(result, expected)
    assert 0 <= result <= dense + 1e-12
    actual_gradient = torch.autograd.grad(result, logits, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert torch.autograd.gradcheck(lambda value: coarse_kl(value.softmax(-1)[selected][None], p[selected][None],
        reverse=reverse, epsilon=1e-12), (logits,))


def test_tail_mass_is_not_discarded_by_topk_renormalization():
    teacher = torch.tensor([[.2, .3]])
    student = torch.tensor([[.1, .15]], requires_grad=True)
    loss = coarse_kl(student, teacher, reverse=False, epsilon=1e-8)
    assert loss.item() > .1
    loss.sum().backward()
    assert student.grad.abs().sum() > 0


@pytest.mark.parametrize("reverse", [False, True])
def test_all_vocab_and_zero_tail_match_full_kl(reverse):
    q = torch.tensor([[.2, .3, .5]], requires_grad=True)
    p = torch.tensor([[.4, .4, .2]])
    actual = coarse_kl(q, p, reverse=reverse, epsilon=1e-8)
    expected = (q * (q.log() - p.log())).sum(-1) if reverse else (p * (p.log() - q.log())).sum(-1)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(q.grad).all()


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("projection_scale", [1., 1e-12])
def test_cross_tokenizer_chunk_gradient_matches_independent_reference(tmp_path, reverse, device, projection_scale):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    path = str(tmp_path / "projection.pt")
    torch.save({"indices": torch.tensor([[0, 1], [2, 3], [-1, -1]]),
                "likelihoods": torch.tensor([[7., 3.], [2., 8.], [0., 0.]]) * projection_scale}, path)
    cfg = config(path)
    cfg["reverse_kl"] = reverse
    torch.manual_seed(7)
    logits = torch.randn(2, 5, 3, device=device, requires_grad=True)
    ids = torch.full((2, 7, 2), -1, dtype=torch.long, device=device)
    lp = torch.full((2, 7, 2), -float("inf"), device=device)
    present = torch.zeros((2, 7, 2), dtype=torch.bool, device=device)
    distributions = torch.tensor([[.4, .1, .3, .2], [.1, .3, .2, .4], [.3, .2, .4, .1]], device=device)
    choices = torch.tensor([[0, 2], [1, 3], [2, 0]], device=device)
    ids[:, 2:5] = choices
    lp[:, 2:5] = distributions.gather(-1, choices).log()
    present[:, 2:5] = True
    data = BatchedDataDict(input_ids=torch.zeros(2, 5, dtype=torch.long, device=device),
                          token_mask=torch.tensor([[0, 0, 1, 1, 0]] * 2, device=device),
                          sample_mask=torch.tensor([1., 0.], device=device),
                          teacher_0_topk_ids=ids, teacher_0_topk_logprobs=lp, teacher_0_topk_valid=present,
                          teacher_0_token_mask=torch.tensor([[0, 0, 0, 1, 1, 1, 0]] * 2, device=device),
                          alignment_0_student_chunk_id=torch.tensor([[-1, -1, 0, 0, -1]] * 2, device=device),
                          alignment_0_teacher_chunk_id=torch.tensor([[-1, -1, -1, 0, 0, 0, -1]] * 2, device=device),
                          alignment_0_pair_valid=torch.ones(2, 1, dtype=torch.bool, device=device),
                          alignment_0_pair_is_correct=torch.ones(2, 1, dtype=torch.bool, device=device))
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        actual, metrics = ProjectedTopKTailLoss(cfg)(logits, data, torch.tensor(1., device=device), torch.tensor(2., device=device))
    ref_logits = logits.detach().clone().requires_grad_(True)
    matrix = torch.tensor([[.7, .3, 0., 0.], [0., 0., .2, .8], [0., 0., 0., 0.]], device=device)
    projected = ref_logits[0, 1:3].softmax(-1).mean(0) @ matrix
    expected = torch.stack([reference(projected, distributions[index], choices[index], reverse) for index in range(3)]).mean()
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(logits.grad, ref_logits.grad, rtol=2e-5, atol=2e-6)
    assert logits.grad[1].abs().sum() == 0
    assert logits.grad[0, 0].abs().sum() == 0
    assert logits.grad[0, 1:3].abs().sum() > 0
    assert metrics["valid_teacher_positions_t0"] == 3


@pytest.mark.parametrize("field,value", [("temperature", 2.), ("gold_loss", True), ("xtoken_loss", True),
                                       ("ce_loss_scale", .1), ("kd_loss_mode", "averaged_logits")])
def test_rejects_incompatible_objective_options(field, value):
    cfg = config(None)
    cfg[field] = value
    with pytest.raises(ValueError):
        ProjectedTopKTailLoss(cfg)


def test_real_http_server_limited_to_twenty_preserves_extra_sampled_token():
    payloads = []
    full = torch.linspace(-2, 2, 32).log_softmax(-1)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            payloads.append(payload)
            if payload["prompt_logprobs"] != 20:
                self.send_response(400)
                self.end_headers()
                return
            choices = []
            for index, prompt in enumerate(payload["prompt"]):
                rows = [None]
                for token in prompt[1:]:
                    selected = list(range(12, 32))
                    if token not in selected:
                        selected.append(token)
                    rows.append({str(i): {"logprob": full[i].item()} for i in selected})
                choices.append({"index": index, "prompt_token_ids": prompt, "prompt_logprobs": rows})
            body = json.dumps({"choices": choices}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    cfg = {"top_k": 20, "model_vocab_size": 32, "api_key_env": None, "max_score_elements": 1000,
           "max_response_bytes": 100000, "request_timeout_s": 2, "max_attempts": 1,
           "retry_backoff_s": 0, "normalization_atol": .001}
    client = XTokenTopKLogprobClient(cfg)
    try:
        scores = asyncio.run(client.score_group({"url": "http://127.0.0.1:" + str(server.server_port), "model": "fixture"},
                                               torch.tensor([[1, 2, 3]]), torch.tensor([3])))
        assert scores["topk_valid"][0, 0].sum() == 21
        assert 2 in scores["topk_ids"][0, 0]
        assert scores["topk_valid"][0, 2].sum() == 0
        assert scores["topk_logprobs"][0, 0].exp().sum() < 1
        assert payloads[0]["prompt_logprobs"] == 20 and payloads[0]["logprobs"] is None
        with pytest.raises(TeacherCapabilityError):
            client._parse_topk_row({"0": {"logprob": 0.}})
    finally:
        asyncio.run(client.close())
        server.shutdown()
        server.server_close()
        thread.join()
