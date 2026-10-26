"""Create tiny local fixtures and run the actual off-policy or OPD entrypoint."""

import argparse
from pathlib import Path
import subprocess
import sys

from datasets import Dataset
from omegaconf import OmegaConf
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


def create_model(path, *, teacher, weak_distribution=None, architecture=None):
    """Offline fixture with different token IDs and distinct teacher probabilities."""
    path.mkdir(parents=True, exist_ok=True)
    words = ["<pad>", "<unk>", "<eos>", "a", "b", "c", *[f"w{index}" for index in range(6, 32)]]
    if teacher:
        words[3:6] = ["c", "a", "b"]
    vocab = {word: index for index, word in enumerate(words)}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",
                                        unk_token="<unk>", eos_token="<eos>")
    tokenizer.chat_template = ("{% for message in messages %}b {{ message['content'] }} a {% endfor %}c "
                                if teacher else "{% for message in messages %}a {{ message['content'] }} b {% endfor %}c ")
    tokenizer.save_pretrained(path)
    torch.manual_seed(23)
    model_class, config_class = (Qwen3ForCausalLM, Qwen3Config) if architecture == "qwen3" else (LlamaForCausalLM, LlamaConfig)
    model = model_class(config_class(vocab_size=32, hidden_size=128, intermediate_size=256,
                       num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                       max_position_embeddings=64, pad_token_id=0, eos_token_id=2,
                       tie_word_embeddings=architecture == "qwen3"))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for name, parameter in model.named_parameters():
            if name.endswith("norm.weight"):
                parameter.fill_(1)
        model.model.embed_tokens.weight.fill_(0.1)
        model.lm_head.weight.fill_(-0.1)
        strengths = [0.13, 0.11, 0.12] if teacher else [0.11, 0.13, 0.12]
        for word, strength in zip(["a", "b", "c"], strengths):
            model.lm_head.weight[vocab[word]].fill_(strength)
        if weak_distribution:
            model.lm_head.weight.zero_()
            model.lm_head.weight[:3].fill_(-.1)
            for word, strength in zip(["a", "b", "c"], strengths):
                model.lm_head.weight[vocab[word]].fill_(strength / 10)
    model.save_pretrained(path)
    return vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["off", "on"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote-url", type=str, default=None)
    parser.add_argument("--score-mode", choices=["full", "topk"], default=None)
    parser.add_argument("--student-architecture", choices=["llama", "qwen3"], default=None)
    args = parser.parse_args()
    root = args.output
    student = create_model(root / "student", teacher=False, weak_distribution=args.score_mode == "topk", architecture=args.student_architecture)
    teacher = create_model(root / "teacher", teacher=True, weak_distribution=args.score_mode == "topk")
    projection = root / "projection.pt"
    torch.save({"indices": torch.tensor([[teacher[word]] for word in student]),
                "likelihoods": torch.ones(32, 1)}, projection)
    data_path = root / "prompts.parquet"
    Dataset.from_dict({"text": ["a b c"] * 8}).to_parquet(data_path)
    register_omegaconf_resolvers()
    recipe = "xtoken_topk_" if args.score_mode == "topk" else "xtoken_remote_" if args.remote_url else "xtoken_"
    cfg = OmegaConf.to_container(load_config(f"examples/configs/{recipe}{args.mode}_policy_distillation.yaml"), resolve=True)
    cfg["cluster"] = {"num_nodes": 1, "gpus_per_node": torch.cuda.device_count()}
    cfg["distillation"].update({"num_prompts_per_step": 2, "max_num_steps": 2, "max_num_epochs": 1})
    cfg["data"]["num_workers"] = 0
    cfg["data"]["train"].update({"data_files": str(data_path), "characters_per_sample": None})
    cfg["logger"].update({"log_dir": str(root / "logs"), "monitor_gpus": False})
    cfg["checkpointing"]["enabled"] = False
    for index, policy in enumerate([cfg["policy"], cfg["teachers"][0]]):
        path = str(root / ("student" if index == 0 else "teacher"))
        policy["model_name"] = path
        policy["tokenizer"]["name"] = path
        policy["train_global_batch_size"] = 2
        policy["max_total_sequence_length"] = 32
        policy["make_sequence_length_divisible_by"] = 4
        if "dtensor_cfg" in policy:
            policy["dtensor_cfg"].update({"activation_checkpointing": False, "expert_parallel_size": 1})
            policy["optimizer"]["kwargs"]["lr"] = .01
            policy["scheduler"] = {"name": "torch.optim.lr_scheduler.ConstantLR", "kwargs": {"factor": 1., "total_iters": 10}}
    if args.remote_url:
        remote = cfg["teachers"][0]["remote"]
        remote.update({"url": args.remote_url, "model": "xtoken-fixture", "model_vocab_size": 32,
                       "max_score_elements": 4096, "max_response_bytes": 1048576,
                       "api_key_env": "VLLM_API_KEY"})
        cfg["loss_fn"]["vocab_topk"] = 32
    cfg["teachers"][0]["projection_matrix_path"] = str(projection)
    cfg["policy"]["generation"]["vllm_cfg"].update({"async_engine": False, "skip_tokenizer_init": True,
                        "gpu_memory_utilization": 0.05, "max_model_len": 32, "enforce_eager": True})
    cfg["policy"]["generation"]["max_new_tokens"] = 4
    cfg["policy"]["generation"]["vllm_kwargs"] = {"max_num_seqs": 4, "max_num_batched_tokens": 128}
    if args.mode == "on":
        cfg["on_policy"]["max_prompt_tokens"] = 16
        cfg["on_policy"]["refit_buffer_size_gb"] = 0.125
    config_path = root / "effective.yaml"
    OmegaConf.save(OmegaConf.create(cfg), config_path)
    entrypoint = f"examples.run_xtoken_{args.mode}_policy_distillation"
    result = subprocess.run([sys.executable, "-m", entrypoint, "--config", str(config_path)], check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
