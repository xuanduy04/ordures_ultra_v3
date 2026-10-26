# X-Token qua NeMo Gym trên fork Super

Contract G1: Qwen3-1.7B sinh một response qua NeMo Gym `general_qa`, teacher
Gemma-4-26B-A4B-it chấm response bằng tokenizer riêng. Objective vẫn là
`topk_tail_kl`, K=20 tại mỗi vị trí + OTHER, CE=0, T=1. Đây là adaptation;
không gọi nó là P-KL nguyên bản, PPO hay sampled-token MOPD của GRPO.

## Nguồn và thư mục

Mọi source mới, data dẫn xuất, assets và run output thuộc
`/workspace/storage-shared/nlp/tungks/xtoken_all/`.
Không ghi vào checkout `/workspace/storage-shared/nlp/vai-1.4-30b/nemo-rl/`
hoặc `/opt/nemo-rl`; tái dùng interpreter, venv và dependency đã provision.
Không sửa/restart server teacher.

Nhánh `tungks/feat/xtoken-all-super` kế thừa X-Token `70534ca` và delta từ
bốn tarball công ty ngày 2026-10-26. Xem
`tools/xtoken_all/company-artifacts.json` để đối chiếu nguồn gốc từng gói.
Các delta Gym/GRPO/penalty và test của công ty được giữ cùng adapter mới.
Delta `custom_utils` được giữ nguyên theo gói, không refactor converter gốc.

## Luồng thực thi

1. `NemoGymDataset` nạp JSONL; `NemoGymPromptCollator` giữ `extra_env_info` thật,
   không đọc placeholder `message_log` của processor GRPO.
2. Refit student hiện tại sang các worker vLLM; chỉ sau khi đồng bộ mới gửi
   batch cho Gym. Engine dùng async HTTP để phục vụ Gym, nhưng lịch cập nhật
   X-Token là đồng bộ: không dùng buffer rollout cũ.
3. `general_qa_simple_agent_reasoning_off` sinh một response. Adapter kiểm tra
   prompt IDs giống template Qwen và giữ nguyên sampled response IDs.
4. Hoàn tất batch rồi cho generation ngủ để policy dùng lại GPU. Giới hạn
   thời gian bao gồm cả chờ stream Ray, không chỉ `ray.get` kết quả đã sẵn sàng.
5. Collator cũ tokenize cho Gemma và alignment; client cũ lấy teacher scores;
   loss projection/top-20/OTHER cũ cập nhật student.
6. Một batch chỉ được dùng cho một update. Hai lượt smoke phải có policy
   version 0, 1; chỉ tăng version khi update thành công. Reward QA chỉ ghi log.

G1 bật `retain_truncated_token_ids`: response chạm output budget vẫn giữ prefix
đã sinh để KD, nếu Gym trả đủ token IDs. Mẫu bị từ chối trước khi sinh hoặc thiếu
token IDs vẫn bị mask. GRPO cũ không bật cờ này và giữ masking của công ty.

## Dữ liệu

`tools/xtoken_all/convert_general_qa.py` gọi lại `_convert_entry` của converter
công ty. Truyền rõ input, output mới, expected rows, prompt field, answer field
và dataset name. Hỗ trợ JSONL/Parquet, field lồng dạng `reward_model.ground_truth`.
Output gồm `train.jsonl`, `source-metadata.jsonl`, `conversion.json` có SHA256.
Không resample/dedup/filter; lỗi để lại `FAILED`, không tạo receipt thành công.

DAPO r2 đã chuẩn hóa: dùng `data/dapo/train.jsonl`, 16.367 dòng,
prompt field `text`, answer field `ground_truth`. Không trộn validation/test.
Raw DAPO Parquet có schema khác và nhiều dòng lặp; không khai 16.367 cho file raw.
Pool `math_code_20k` chưa được đọc thực tế: không đưa toàn pool vào verifier QA.
CodeForces cần execution verifier; converter từ chối các source code đã nhận biết.
Nguồn không có metadata task vẫn cần operator kiểm tra bản chất bài toán.

Smoke 128 prompt tokens + 32 response tokens không phù hợp mọi mẫu DAPO.
`make_smoke.py` tạo một bộ 8 mẫu đầu tiên thỏa budget, giữ thứ tự, ghi chỉ số
dòng nguồn/token count và hash; không sửa pool train. Không coi smoke này là
tập đánh giá chất lượng. Sau smoke phải chốt budget và dataset pilot riêng.

## Cấu hình và runtime Gym

Điểm vào: `examples.run_xtoken_on_policy_distillation` với config được render
từ `examples/configs/xtoken_gym_qwen3_gemma4.yaml`. Renderer bắt buộc source,
data, projection và run directory nằm dưới `XTOKEN_ALL_ROOT`, chọn đúng kiến trúc
Qwen3-1.7B. Smoke: 4 GPU 0–3, TP=CP=1, GBS=4/MBS=1, 2 updates, checkpoint tắt.
Những số này không phải cấu hình mặc định cho full training.

`render.py` nhận `--gym-python`, `--gym-venvs`, `--gym-source` trỏ các thành phần
đã provision. Gym import phải đúng source đã khai; config paths dùng đường dẫn
tuyệt đối. Adapter bật `skip_load_from_dotenv=true`: không cần copy/nạp
`nemo_gym_env.yaml`, không đưa secret của pipeline khác vào effective config.

Gym fork `8949f55` luôn validate endpoint judge lúc khởi động, kể cả khi
`should_use_judge=false`. Config G1 khai endpoint/model của teacher đã có cho
phép kiểm tra metadata; không gọi LLM judge để chấm đáp án. Nếu endpoint cần
auth mà validator Gym không hỗ trợ, dừng và báo đúng lỗi, không mở server mới.
Live Gym fork khác commit hoặc dirty thì cần so diff trước khi dùng kết luận này.

## Chuỗi helper cho agent vận hành

Tất cả helper nằm tại `tools/xtoken_all/`; dùng interpreter công ty đã có.
Không bootstrap dependency hay bật Docker local.

1. Đối chiếu active source với tarball, tạo source mới dưới `xtoken_all/src`;
   giữ source gốc. Nếu thiếu bundled dependency, liên kết rõ thư mục 3rdparty
   đã có hoặc dùng PYTHONPATH đã provision, không tự clone/cài lại.
2. Chuyển DAPO bằng `convert_general_qa.py`, tạo smoke bằng `make_smoke.py`.
3. Render bằng `render.py`: tham số đầy đủ có trong `--help`; mọi path bắt buộc.
   `provenance.json` chốt hash code, data, projection và effective YAML.
4. Chạy `preflight.py --run-dir <run>` bằng `NRL_PYTHON`. Nó kiểm tra shard,
   tokenizer, projection qua loader thật của X-Token, budget từng prompt,
   source import của hai worker và config/venv Gym. Nó không khởi động server.
5. `probe_teacher.py --config <effective.yaml> --server-mode-evidence <json>`
   kiểm alias, ba ca tokenizer parity, một yêu cầu score nhỏ và cap top-20.
   Evidence JSON cần `teacher_url`, `served_model`, `logprobs_mode=raw_logprobs`,
   `verification_source` là `operator_checked_server_args` hoặc
   `server_reported_config`, `observed_at_utc`, `evidence_description` đã bỏ secret.
   Phải kiểm tra cấu hình server thật; probability mass/echo không chứng minh mode.
6. Operator xác nhận node, bốn UUID và Ray ownership hiện tại. Sau đó dùng
   `check_allocation.py` với `--node-label`, bốn `--expected-gpu-uuids`,
   `--ray-address`, `--operator-owned-ray-verified`, `--output` mới.
   Helper không start/stop Ray. Nó đòi 1 Ray node/4 GPU và GPU mục tiêu không bận.
7. Export `XTOKEN_ALL_ROOT`, `XTOKEN_RUN_DIR`, `NRL_PYTHON`,
   `NEMO_RL_VENV_DIR`, `XTOKEN_ALLOCATION_RECEIPT`; chạy `launch.sh`.
   Launcher kiểm UUID/occupancy lại ngay trước chạy, receipt chưa quá 5 phút,
   hash source/data/projection/config; không `ray stop --force`.
8. `check_result.py --run-dir <run>` đòi exit driver và tee bằng 0, đúng hai
   update, loss/gradient hữu hạn và gradient khác 0, teacher scores đúng dạng,
   Gym rollout versions 0 rồi 1, response không rỗng, reward không vào loss.

## Kiểm chứng và giới hạn

Test nhẹ độc lập tensor: `tools/xtoken_all/tests/test_gym_contract.py`.
Test CPU tensor/Ray mock: `tests/unit/algorithms/x_token/test_gym_rollout.py`.
Chạy thêm regression X-Token đang có và các test company processor/Gym/penalty
trong runtime công ty. Không gộp nhiều test suite resource server Gym có cùng
tên `test_app.py` vào một lệnh.

Test protocol/converter không chứng minh server HTTP, CUDA IPC, FSDP refit hay
GPU training. Chỉ sau khi vượt preflight và hai update trên node mới báo smoke
GPU pass; vẫn chưa có kết luận chất lượng, hội tụ, resume parity hoặc full train.
Same-tokenizer Qwen235B là nhánh so sánh sau; không đổi loss/backend giữa hai nhánh.
