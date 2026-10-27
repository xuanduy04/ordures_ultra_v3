# DSH — r10: student decode guard và export checkpoint A-on-11 để eval

## Bằng chứng mới

A-on-11 hoàn tất step66: loss1.4525964260101318, grad_norm1.8180427551269531,
lr1.374e-05. Step67 dừng ở collator khi decode sampled student IDs ra U+FFFD,
trước teacher scoring; không phải bằng chứng Gemma sinh text hỏng. Log không
phân biệt được byte UTF-8 dang dở ở giới hạn response với U+FFFD có sẵn trong
output. r10 xử lý bảo thủ, không khẳng định cơ chế byte cụ thể đã được đo.

Checkpoint được liệt kê: A-on-11/step_20,30,50,60. A-on-10 chỉ có step_1,2;
không có A-on-10/step_20. Giữ các mốc đã có; không sửa config/weights gốc.

## Thay đổi training

- Cross-tokenizer: nếu student decode chứa U+FFFD, giữ nguyên toàn bộ sampled
  IDs, nhưng sample_mask/token_mask/alignment view của mẫu đó bằng0. Teacher
  không nhận text U+FFFD; các mẫu hợp lệ còn lại vẫn train đúng batch/mẫu gốc.
- Có log XTOKEN_RESPONSE_MASK với policy_version, rollout_id, source_index,
  số response token và lý do; không ghi prompt hay response text.
- Gym valid_samples/reward metrics dùng mask sau collation. Theo dõi tỷ lệ
  mask; không gọi đây là data đầy đủ không-filter sau generation.
- Same-tokenizer không cần decode để retokenize: dùng sampled IDs trực tiếp.
- Nếu toàn batch không còn response hợp lệ, dừng rõ ràng trước teacher forward
  và optimizer; không tạo update0, không tự dùng lại rollout hoặc tăng version.
- Lỗi structural/prompt mismatch/control token/overflow vẫn dừng, không catch
  mọi ValueError để che lỗi. Không repair U+FFFD hoặc cắt token rồi pretend parity.

Đây là thay đổi contract masking có log, không tự resume A-on-11 lịch cũ.
Chạy pilot mới theo `HANDOFF-pilot-200.md`, ưu tiên B trên anhhbt. Source mới
`ordures_ultra_v3_pilot200_r10`, steps200/warmup20/save10/data A-on-11 không đổi.
Không ghi đè source của job đang chạy. Ray B đã thấy172.29.167.243:6379 và QA_APP_OK.
Teacher B đã thấy Qwen3-32B, version0.14.2.dev0+gd7de043d5.d20260219; mode evidence
vẫn cần probe xác nhận, không restart hay sửa server.

## Export checkpoint thấp nhất hiện có: step20

Không đổi tên shard. Super Automodel có hàm
`nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors.consolidate_safetensors_files`.
Nó đọc saved_offsets trong header để ghép các mảnh tensor; utility mới gọi chính
hàm runtime này. `convert_dcp_to_hf.py` cũ đọc .metadata/.distcp, không phù hợp
để đoán cách ghép các shard safetensors đang thấy.

Metadata thường nằm ở `policy/weights/model/.hf_metadata/`: config.json,
generation_config.json và fqn_to_file_index_mapping.json. `ls` cũ không có -a
nên chưa kết luận chúng thiếu. Utility inspect đọc header, không nạp model/GPU.

Block R10-E01 — chạy trên node rl-test-opd-ultra-selfbuilt-trainllm-1-0-1 — inspect checkpoint20, không sửa nguồn

```bash
(
ROOT=/workspace/storage-shared/nlp/tungks/xtoken_all
SOURCE="$ROOT/ordures_ultra_v3_pilot200_r10"
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python
CKPT="$ROOT/runs/A-on-11/checkpoints/step_20"
export PYTHONPATH="$SOURCE${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 UV_OFFLINE=1
"$PY" -B "$SOURCE/tools/xtoken_all/export_checkpoint.py" --checkpoint "$CKPT" --inspect
STATUS=$?
echo "R10_E01_EXIT_STATUS=$STATUS"
echo "--- het Block R10-E01, shell con song ---"
exit "$STATUS"
)
```

Nếu mapping metadata thiếu: dừng, gửi output inspect và listing ẩn. Không sinh
mapping bằng đoán tên file. Nếu chỉ config thiếu, có CLI --base-model tới model
gốc sau khi đối chiếu config train; fallback ấy được ghi rõ trong receipt.

Block R10-E02 — chạy trên node rl-test-opd-ultra-selfbuilt-trainllm-1-0-1 — consolidate vào thư mục mới và kiểm load/forward HF trên CPU

```bash
(
ROOT=/workspace/storage-shared/nlp/tungks/xtoken_all
SOURCE="$ROOT/ordures_ultra_v3_pilot200_r10"
PY=/opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python
CKPT="$ROOT/runs/A-on-11/checkpoints/step_20"
OUTPUT="$ROOT/exports/A-on-11-step20-hf-r10"
export PYTHONPATH="$SOURCE${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 UV_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
"$PY" -B "$SOURCE/tools/xtoken_all/export_checkpoint.py" --checkpoint "$CKPT" \
  --output-dir "$OUTPUT" --validate-load
STATUS=$?
echo "R10_E02_EXIT_STATUS=$STATUS"
echo "--- het Block R10-E02, shell con song ---"
exit "$STATUS"
)
```

CPU export dùng runtime policy đã có, không tải/cài, không khởi tạo Ray/NCCL,
không load optimizer. Cần disk mới cỡ checkpoint weights, thêm RAM cho tensor
lớn nhất và load Qwen3-1.7B CPU khi validate; source không bị ghi. Hàm consolidate
chạy1 thread theo YAML. Các shard nguồn được hash trước/sau; config/tokenizer
được lấy từ chính checkpoint. Output tồn tại thì từ chối ghi đè; nếu lỗi để FAILED,
không có export-receipt thành công. Không serve output FAILED.

Khi E02 success=true và hf_load_validation.success=true, đường HF để đưa cho
agent eval là `/workspace/storage-shared/nlp/tungks/xtoken_all/exports/A-on-11-step20-hf-r10`.
Đó là model path và tokenizer path cho vLLM; không dùng root checkpoint NeMo.
Giữ lệnh serve đã chạy của operator, chỉ đổi hai path sau khi xác nhận node/GPU/port
eval rảnh. Gói này không kill PID hoặc deploy teacher/eval server tự động.

Load/finite-forward chưa chứng minh logits parity với policy distributed gốc
hoặc model đã cải thiện. Eval metric/benchmark là bước riêng. Chưa có output E01/E02
trên node để tuyên bố checkpoint thật đã convert/load thành công.

## Kiểm chứng trong gói

Offline regression dùng collator thật với tensor doubles: mixed bad/good giữ IDs,
mask riêng sample hỏng, same-tokenizer không decode, all-invalid không update.
Test exporter chạy đúng thuật toán byte-copy trong source Automodel công ty trên
hai shard nhỏ có offset, đọc lại safetensors và so sánh chính xác tensor số.
Đó không phải GPU test hoặc parity của checkpoint thật; test Torch đầy đủ cũng
được thêm vào tests/unit để chạy trên runtime node sẵn có.
