# X-Token U7 trong OPD

`on_policy_distillation.mode = "xtoken_topk"` chọn loss projected top-k + OTHER của U7 trong entrypoint hiện có `examples/nemo_gym/run_grpo_nemo_gym.py`. Không cần entrypoint X-Token riêng. Khi bỏ `mode`, OPD tiếp tục dùng `sampled_token`.

## Phạm vi hỗ trợ

- Một teacher URL/model; nhiều agent alias có thể trỏ cùng cặp đó.
- NeMo-Gym, một user text và một assistant response; không tools, history, system instructions hay multimodal.
- DTensor V2, TP=1, CP=1; fixed microbatches; không sequence packing.
- Non-colocated vLLM async engine, nhưng dùng hàng rào giữa các phiên bản student: `max_trajectory_age_steps=0`, không in-flight weight updates. Có thể mất throughput so với OPD async có staleness.
- Một optimizer step trên toàn bộ batch rollout; `train_global_batch_size = num_prompts_per_step * num_generations_per_prompt`. Microbatch vẫn có gradient accumulation.
- Student sampling: temperature=1, top_p=1, không generation top-k. Teacher scoring luôn ở temperature=1.
- KD-only: không sampled-token OPD advantages, PPO/ICE-POP importance ratio, ORM loss, reference-policy KL hay reward shaping. Reward Gym chỉ là metric. Không có khẳng định đây là Gold/H-KL hoặc full-vocabulary KL.

## Cấu hình Python

Các trường dưới đây phải có trong `master_config` trước `grpo.setup()`. Đây là phần bổ sung vào cấu hình OPD hiện có; các trường policy, cluster, dữ liệu và logging khác vẫn do framework quản lý.

```python
master_config["on_policy_distillation"]["enabled"] = True
master_config["on_policy_distillation"]["mode"] = "xtoken_topk"
master_config["on_policy_distillation"]["teacher_model_by_agent_name"] = {
    "your_agent_name": {"url": "http://your-teacher:8000", "model": "served-model-name"},
}
master_config["on_policy_distillation"]["xtoken"] = {
    "teacher_tokenizer": {"name": "/models/teacher-tokenizer", "chat_template": "default"},
    "projection_matrix_path": "/artifacts/student-to-teacher-projection.pt",
    "alignment_max_combination_length": 4,
    "teacher_max_sequence_length": 8192,
    "teacher_micro_batch_size": 1,
    "kl_loss_weight": 1.0,
    "reverse_kl": False,
    "tail_epsilon": 1e-8,
    "remote": {
        "model_vocab_size": 262144,  # Phải khớp teacher đang serve.
        "top_k": 20,
        "max_score_elements": 50_000_000,
        "max_response_bytes": 64_000_000,
        "request_timeout_s": 120.0,
        "max_attempts": 3,
        "retry_backoff_s": 1.0,
        "normalization_atol": 0.001,
        "api_key_env": None,  # Hoặc tên biến môi trường; không đặt secret vào config.
    },
}
master_config["grpo"]["async_grpo"]["max_trajectory_age_steps"] = 0
master_config["grpo"]["async_grpo"]["in_flight_weight_updates"] = False
master_config["policy"]["generation"]["temperature"] = 1.0
master_config["policy"]["generation"]["top_p"] = 1.0
master_config["policy"]["generation"]["top_k"] = None
master_config["loss_fn"]["reference_policy_kl_penalty"] = 0.0
master_config["grpo"]["reward_shaping"]["enabled"] = False
```

Các con số về vocabulary, context, microbatch và giới hạn bộ nhớ là ví dụ, cần chọn theo model và batch thực tế. `max_score_elements` chặn cả tensor top-k của global batch lẫn từng request; không chỉ kích thước HTTP. `api_key_env` được dùng cho cả health probe và scoring.

Teacher tokenizer phải là fast tokenizer có offset mappings và đúng ID ordering của teacher endpoint. Projection phải được xây đúng chiều student → teacher, đúng tokenizer và vocabularies. Framework kiểm tra kích thước/giá trị; không thể suy ra danh tính tokenizer chỉ từ shape projection. Các artifact phải sẵn ở driver và worker cần dùng chúng.

## Luồng dữ liệu và file sở hữu

1. `nemo_rl/algorithms/opd_outsource.py`: đọc mode, resolve teacher theo agent alias, kiểm tra serve. Mode không có hoặc OPD tắt vẫn giữ hành vi cũ.
2. `nemo_rl/algorithms/grpo.py`: chọn `ProjectedTopKTailLoss`, refit generation trước rollout đầu tiên, nhận batch từ replay buffer. Với X-Token, bỏ nhánh sampled logprobs/advantage/penalty và gọi batch builder.
3. `nemo_rl/algorithms/async_utils.py`: X-Token chỉ reserve target bằng phiên bản generation hiện tại. Chỉ refit sau khi rollout đã xong, reset prefix/KV cache thành công rồi mới cho generation tiếp tục. Replay sampling dùng age=0; driver từ chối batch stale.
4. `nemo_rl/algorithms/x_token/opd_integration.py`: kiểm tra single-turn contract và prompt IDs. Dùng token IDs Gym đã sinh, không encode lại response student. Tạo batch cho collator; route một teacher; score theo microbatch có giới hạn; đóng HTTP client cả khi lỗi.
5. `nemo_rl/data/xtoken_rollout_collate.py` và `xtoken_response_text.py`: dựng teacher chat prefix bằng tokenizer teacher, ghép text response, lấy offset để loại token straddle ranh giới prefix/response. Align phần response bằng `TokenAligner`; đưa chunk IDs về tọa độ sequence đầy đủ. Prompt, padding, EOS cuối và structural control tokens không tạo loss. `<think>`/`</think>` được giữ như U7. Sample không an toàn bị mask; nếu cả batch không còn response/alignment hợp lệ thì dừng trước optimizer.
6. `nemo_rl/algorithms/x_token/remote_teacher.py`: gửi teacher IDs với `echo=true`, `max_tokens=0`, `prompt_logprobs=top_k`, numeric token IDs. Kiểm tra echo, choice index, số lượng scores, vocabulary, probability mass, timeout, retry count và response bytes. API row `i` được chuyển sang predictor row `i-1`; hàng cuối không dự đoán target nào.
7. `nemo_rl/algorithms/x_token/topk_tail.py`: logits student chưa chia sampling temperature → FP32 softmax → trung bình xác suất theo student chunk → sparse projection sang teacher vocabulary → các bucket token được teacher trả về và reachable → bucket OTHER. Teacher positions được trung bình trong chunk; chunk được cân theo số student response tokens. `reverse_kl=False` là KL(teacher || projected student); `True` đổi chiều trên cùng partition.
8. `nemo_rl/models/policy/lm_policy.py`, `nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py` và `nemo_rl/models/automodel/data.py`: truyền danh sách bỏ qua kiểm tra student sequence length cho teacher axes và pair axes. Student chunk/mask axes vẫn phải khớp student sequence. Fixed microbatch giữ nguyên các axes này khi shard.
9. `nemo_rl/models/automodel/train.py`: loss có `requires_raw_logits=True` nhận raw logits. Loss cũ và inference post-processors giữ quy tắc temperature trước đó. Denominator là tổng student tokens có sample mask trên global batch, dùng chung cho các microbatch/DP rank.

`advantages` bằng zero trong X-Token batch chỉ là compatibility grid cho phần bookkeeping của driver; loss không đọc nó, và driver không log nó như advantage đo được. Không có sampled-token importance correction trong objective mới.

## Resume và lỗi

Checkpoint replay của X-Token chỉ được chấp nhận khi mọi group có generation version và target version bằng step đang restore. Replay cũ từ async sampled-token mode có stale/future groups bị từ chối, không tự đổi version hoặc bỏ dữ liệu. Cơ chế dataloader/checkpoint vẫn thuộc native async driver; chưa chứng minh resume khôi phục bit-exact thứ tự prompt khi có prefetch/in-flight collection.

Lỗi teacher, hết valid alignment, cache invalidation thất bại và collector lỗi đều phải được xử lý trước bước update tiếp theo. Driver re-raise lỗi để tiến trình gọi không nhận một lần chạy thành công giả.

## Kiểm chứng

Các test mới ở `tests/unit/algorithms/x_token/` chạy được với CPU: independent KL/gradient reference, top-k tail mass, HTTP fixture, byte/retry/timeout limits, token alignment, sampled-ID preservation, masking, different sequence axes, global-token normalization và mode validation.

`test_native_opd_loop.py` thực thi nguyên hàm driver lấy từ source qua AST trong hai bước cho cả hai mode, với Ray actors/model/refit/logging giả lập. Phần batch, alignment, loss và autograd dùng implementation thật. Cách này kiểm tra điều phối Python; không chứng minh NCCL, FSDP, CUDA sparse backward, vLLM refit hay server production hoạt động.

Trên runtime CPU đã có dependency, chạy các test CPU phù hợp bằng `python -m pytest tests/unit/algorithms/x_token/ tests/unit/algorithms/test_opd_outsource.py -q --timeout=60`. Các case CUDA tự skip khi không có GPU. Không cần model weights hoặc truy cập teacher thật cho bộ test này.
