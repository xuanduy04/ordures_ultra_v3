# X-Token U7 trong OPD

`on_policy_distillation.mode = "xtoken_topk"` chọn loss projected top-k + OTHER của U7 trong entrypoint hiện có `examples/nemo_gym/run_grpo_nemo_gym.py`. Không cần entrypoint X-Token riêng. Khi bỏ `mode`, OPD tiếp tục dùng `sampled_token`.

## Phạm vi hỗ trợ

- Một teacher URL/model; nhiều agent alias có thể trỏ cùng cặp đó.
- NeMo-Gym, một user text và một assistant response; không tools, history, system instructions hay multimodal.
- Megatron Core đi qua worker và scheduler native của OPD, dùng TP/CP/PP, sequence parallel, dynamic batching hoặc sequence packing từ policy config. Nhánh DTensor V2 được giữ để tương thích; riêng DTensor vẫn giới hạn TP=CP=1 và fixed microbatches.
- Non-colocated vLLM async engine và replay scheduling của OPD. Khi `max_trajectory_age_steps > 0`, dùng scheduling/staleness/in-flight settings hiện có. Khi đặt age=0, chọn strict fresh-rollout mode: chờ refit, cache reset và chỉ nhận rollout cùng phiên bản.
- Global batch và microbatch theo OPD. Riêng age=0 yêu cầu `train_global_batch_size = num_prompts_per_step * num_generations_per_prompt`, tức một optimizer step trên mỗi fresh rollout batch.
- Native async mode dùng student sampling config hiện có. Strict fresh mode age=0 yêu cầu temperature=1, top_p=1, không generation top-k. Distribution KD và teacher scoring luôn ở temperature=1.
- Objective chính là KD; không dùng sampled-token OPD advantages, PPO/ICE-POP importance ratio, ORM loss hoặc reference-policy KL. Reward Gym và reward shaping chỉ phục vụ metrics trong mode này. Các auxiliary losses do native model/optimizer bật vẫn thuộc policy config. Không có khẳng định đây là Gold/H-KL hoặc full-vocabulary KL.

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
master_config["loss_fn"]["reference_policy_kl_penalty"] = 0.0
```

Giữ nguyên `policy.megatron_cfg`, TP/CP/PP, batching và replay settings của OPD. Khi dùng replay age lớn hơn zero, objective KD được tính trên các prefix từ replay; đây không phải strict fresh on-policy U7 và không có tuyên bố importance-corrected/unbiased. Muốn giữ fresh-rollout contract của U7 thì đặt age=0, tắt in-flight updates và dùng sampling temperature=top_p=1.

Các con số về vocabulary, context, microbatch và giới hạn bộ nhớ là ví dụ, cần chọn theo model và batch thực tế. `max_score_elements` chặn cả tensor top-k của global batch lẫn từng request; không chỉ kích thước HTTP. `api_key_env` được dùng cho cả health probe và scoring.

Teacher tokenizer phải là fast tokenizer có offset mappings và đúng ID ordering của teacher endpoint. Projection phải được xây đúng chiều student → teacher, đúng tokenizer và vocabularies. Framework kiểm tra kích thước/giá trị; không thể suy ra danh tính tokenizer chỉ từ shape projection. Các artifact phải sẵn ở driver và worker cần dùng chúng.

## Luồng dữ liệu và file sở hữu

1. `nemo_rl/algorithms/opd_outsource.py`: đọc mode, resolve teacher theo agent alias, kiểm tra serve. Mode không có hoặc OPD tắt vẫn giữ hành vi cũ.
2. `nemo_rl/algorithms/grpo.py`: chọn `ProjectedTopKTailLoss`, refit generation trước rollout đầu tiên, nhận batch từ replay buffer. Với X-Token, bỏ nhánh sampled logprobs/advantage/penalty và gọi batch builder.
3. `nemo_rl/algorithms/async_utils.py`: routing và scheduling theo native async OPD; không gửi student token IDs sang teacher khác tokenizer. Strict age=0 có thêm hàng rào phiên bản và bắt buộc cache reset sau refit. Native replay age>0 giữ scheduling hiện có.
4. `nemo_rl/algorithms/x_token/opd_integration.py`: kiểm tra single-turn contract và prompt IDs. Dùng token IDs Gym đã sinh, không encode lại response student. Tạo batch cho collator; route một teacher; score theo microbatch có giới hạn; đóng HTTP client cả khi lỗi.
5. `nemo_rl/data/xtoken_rollout_collate.py` và `xtoken_response_text.py`: dựng teacher chat prefix bằng tokenizer teacher, ghép text response, lấy offset để loại token straddle ranh giới prefix/response. Align phần response bằng `TokenAligner`; đưa chunk IDs về tọa độ sequence đầy đủ. Prompt, padding, EOS cuối và structural control tokens không tạo loss. `<think>`/`</think>` được giữ như U7. Sample không an toàn bị mask; nếu cả batch không còn response/alignment hợp lệ thì dừng trước optimizer.
6. `nemo_rl/algorithms/x_token/remote_teacher.py`: gửi teacher IDs với `echo=true`, `max_tokens=0`, `prompt_logprobs=top_k`, numeric token IDs. Kiểm tra echo, choice index, số lượng scores, vocabulary, probability mass, timeout, retry count và response bytes. API row `i` được chuyển sang predictor row `i-1`; hàng cuối không dự đoán target nào.
7. `nemo_rl/algorithms/x_token/topk_tail.py`: logits student chưa chia sampling temperature → FP32 softmax → trung bình xác suất theo student chunk → sparse projection sang teacher vocabulary → các bucket token được teacher trả về và reachable → bucket OTHER. Teacher positions được trung bình trong chunk; chunk được cân theo số student response tokens. `reverse_kl=False` là KL(teacher || projected student); `True` đổi chiều trên cùng partition.
8. `nemo_rl/models/policy/lm_policy.py` chọn worker theo policy config. `nemo_rl/models/policy/workers/megatron_policy_worker.py` và `nemo_rl/models/megatron/data.py` truyền skip keys để teacher/pair axes không bị so hoặc cắt theo student length. `nemo_rl/distributed/batched_data_dict.py` giữ các axes đó khi dynamic microbatch truncate. Packed data giữ metadata ở batch gốc; loss tự map student targets vào cu_seqlens/CP layout, không dùng wrapper chung vốn cắt mọi axis theo student.
9. `nemo_rl/models/megatron/train.py`: nhận raw logits và gọi loss với TP/CP groups trên pipeline loss stage. TP softmax loại padded vocabulary IDs, projection dùng phần hàng của shard rồi SUM; chunk sums/counts được SUM qua CP. SUM của projection/chunk dùng backward identity vì mỗi rank có cùng scalar loss; riêng normalizer của TP softmax dùng autograd reduction đầy đủ. Loss khai báo xử lý packing và CP normalization riêng; MCore schedule vẫn giữ cơ chế bù CP/microbatch averaging hiện có. Nhánh DTensor/Automodel tiếp tục hoạt động như trước. Denominator là tổng student tokens có sample mask trên global batch.

`advantages` bằng zero trong X-Token batch chỉ là compatibility grid cho phần bookkeeping của driver; loss không đọc nó, và driver không log nó như advantage đo được. Không có sampled-token importance correction trong objective mới.

## Resume và lỗi

Trong strict age=0, checkpoint replay chỉ được chấp nhận khi mọi group có generation version và target version bằng step đang restore. Với native async age>0, dùng cơ chế restore/filter replay của OPD. Cơ chế dataloader/checkpoint vẫn thuộc native async driver; chưa chứng minh resume khôi phục bit-exact thứ tự prompt khi có prefetch/in-flight collection.

Lỗi teacher, hết valid alignment, cache invalidation thất bại và collector lỗi đều phải được xử lý trước bước update tiếp theo. Driver re-raise lỗi để tiến trình gọi không nhận một lần chạy thành công giả.

## Kiểm chứng

Các test mới ở `tests/unit/algorithms/x_token/` chạy được với CPU: independent KL/gradient reference, top-k tail mass, HTTP fixture, byte/retry/timeout limits, token alignment, sampled-ID preservation, masking, different sequence axes, global-token normalization và mode validation.

`test_megatron_topk.py` đối chiếu loss và local-logit gradients với dense reference bằng Gloo CPU nhiều process: TP/CP, padded vocabulary, chunk băng qua CP ranks, packed samples, PP-style trailing padding và hai chiều KL. Test gọi loss closure thật lấy từ source Megatron, bao gồm schedule scaling. Đây là kiểm chứng tensor algebra/layout trên CPU, không phải model/pipeline/NCCL GPU canary.

`test_native_opd_loop.py` thực thi nguyên hàm driver lấy từ source qua AST trong hai bước cho cả hai mode, với Ray actors/model/refit/logging giả lập. Phần batch, alignment, loss và autograd dùng implementation thật. Cách này kiểm tra điều phối Python; không chứng minh NCCL, FSDP, CUDA sparse backward, vLLM refit hay server production hoạt động.

Trên runtime CPU đã có dependency, chạy các test CPU phù hợp bằng `python -m pytest tests/unit/algorithms/x_token/ tests/unit/algorithms/test_opd_outsource.py -q --timeout=60`. Các case CUDA tự skip khi không có GPU. Không cần model weights hoặc truy cập teacher thật cho bộ test này.


## Native GPU smoke test

Ngày 2026-11-10, smoke test đã chạy trên một node Modal 4 H100 với PyTorch 2.10.0+cu128, Transformer Engine 2.12.0, cuDNN 9.10.2, NCCL 2.27.5 và Megatron Core 0.16.0rc0 bundled trong repo.

Ma trận TP/CP (1,1), (2,1), (1,2), (2,2), mỗi cấu hình với packing tắt/bật và hai chiều KL, đạt 16 optimizer steps. Test dùng GPT BF16 hai layer, hidden size 128, bốn attention heads, student vocabulary 37 (model padded vocabulary 64), teacher vocabulary 29 và top-k=4. Teacher scores/alignment là fixture có kiểm soát; student forward, Transformer Engine attention, NCCL, native microbatch preprocessing/scheduler, Megatron DDP, backward và optimizer SGD đều chạy thật. Không dùng AST extraction hoặc mock trong luồng GPU này.

Mỗi bước kiểm tra token ownership sau CP/packing, loss và local-logit gradients so với công thức dense độc lập, gradient tham số hữu hạn và cập nhật tham số khác zero. Tensor gather chỉ phục vụ reference trong test; production loss vẫn dùng shard/reduction. Assertions chạy trên mọi rank; metric tổng hợp rank 0 ghi nhận sai lệch gradient logits lớn nhất khoảng 2.04e-10.

Hai regression được sửa từ lượt GPU: collective materialize storage contiguous trước NCCL; non-packed preprocessing shard input IDs, position IDs, query axis của attention mask và MTP mask theo CP. Metadata/alignment vẫn giữ full student coordinates để loss map đúng predictor order.

Script tái lập: tests/test_suites/xtoken_megatron_smoke.py. Trong runtime GPU đã có dependency của repo, từ repo root:

~~~sh
PYTHONPATH=3rdparty/Megatron-LM-workspace/Megatron-LM:. python -m torch.distributed.run --standalone --nproc-per-node=4 tests/test_suites/xtoken_megatron_smoke.py --tp 2 --cp 2 --packed -1
~~~

Tham số --packed -1 chạy cả hai layout; 0/1 chọn một layout. Số process phải bằng TP*CP. Script ghi projection fixture nhỏ vào thư mục tạm của runtime.

Runtime kiểm tra phải có một bộ cuDNN nhất quán. Base image có cuDNN 9.8 trong khi Torch wheel yêu cầu 9.10.2 đã gây lỗi version/sublibrary loading; lượt đạt dùng image kiểm tra đã loại system cuDNN cũ khỏi loader paths. Với TE 2.12, đường cuDNN fused THD dùng trong test giới hạn SM90 trở lên; A100/L40S chưa được qualify packing qua attention backend khác. Xem điều kiện backend trong [source Transformer Engine 2.12](https://github.com/NVIDIA/TransformerEngine/blob/v2.12/transformer_engine/common/fused_attn/fused_attn.cpp).

Kết quả này chưa kiểm chứng model Gemma/Qwen thật, B200, PP>1, MoE/FP8, live teacher HTTP, vLLM refit, Ray actor orchestration hoặc campaign training dài. Nó không chứng minh throughput, VRAM hay chất lượng distillation trên workload production.
