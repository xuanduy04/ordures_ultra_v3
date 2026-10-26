# Bổ sung pool cố định 10k math + 10k code cho X-Token/Gym

Áp dụng lên source r8 `46c1e64dd2e39fdf623b298f9d2e03ec7252cc4b` trong checkout
mới hoặc checkout không được job sử dụng. Không sửa source/data của A-on-11
đang chạy, không restart teacher, không rebuild runtime. Patch trong gói bổ sung
có manifest SHA trước/sau; review `git apply --check` ở checkout mới trước khi áp.
Nếu source khác base, review diff thay vì copy đè. Không sửa Gym source.

## Ý nghĩa

- Input được giữ tại đúng path pool Rethinking cũ trong recipe YAML.
- Không resample, shuffle, dedup hoặc filter lúc convert; source_row_index giữ thứ tự.
- Chỉ nhận đúng 10.000 math + 10.000 code theo nhãn/mapping khai rõ trong YAML.
- Recipe giả định field `prompt`, `reward_model.ground_truth`, `ability`,
  `data_source`; đây chưa phải schema được xác nhận từ file trên node.
- `--inspect` chỉ đọc metadata/count/hash, không in prompt/answer/test case.
  Nếu classification_errors khác rỗng, hoặc count/schema không đúng, dừng và
  chỉnh mapping/field trong bản recipe riêng sau khi kiểm schema. Không hạ count
  hoặc đổi nhãn để vượt gate, không tải lại pool thay thế.
- `train.jsonl` là pool train đầy đủ, không tự chia validation hoặc chọn smoke.
  Validation phải là tập held-out riêng; không lấy cùng pool làm đánh giá.
- Code labels/test cases giữ trong source-metadata; expected_answer mang chuỗi
  JSON của label có cấu trúc. Không đưa chúng vào prompt generation.
- QA text score của code bị loại khỏi metric, không ảnh hưởng KD mask/loss.
  Đây là KD-only, chưa có sandbox/execution verifier. Không báo code accuracy.
- Converter lỗi để `FAILED`, không tạo conversion.json thành công. Không dùng
  file train.jsonl còn dở trong thư mục đó.

## Inspect, sau khi overlay đã được review và áp vào source mới

Block M20K-01 — chạy trên node rl-test-opd-ultra-selfbuilt-trainllm-1-0-1 — inspect pool gốc, không convert hoặc launch

```bash
(
ROOT=/workspace/storage-shared/nlp/tungks/xtoken_all
SOURCE="$ROOT/ordures_ultra_v3"
PY=/opt/nemo_rl_venv/bin/python
RECIPE="$ROOT/data/math-code-20k-conversion.yaml"
if [ -f "$SOURCE/tools/xtoken_all/convert_math_code.py" ] && [ -x "$PY" ]; then
  mkdir -p "$ROOT/data"
  if [ ! -e "$RECIPE" ]; then
    cp -n "$SOURCE/tools/xtoken_all/math_code_20k.yaml" "$RECIPE"
  fi
  cd "$SOURCE" && "$PY" -B -m tools.xtoken_all.convert_math_code --config "$RECIPE" --inspect
else
  printf '%s\n' 'STOP: thiếu overlay hoặc interpreter đã provision'
fi
echo "--- het Block M20K-01, shell con song ---"
)
```

## Convert sau khi inspect khớp 10k + 10k và các field

Block M20K-02 — chạy trên node rl-test-opd-ultra-selfbuilt-trainllm-1-0-1 — convert pool vào output mới, giữ nguyên input

```bash
(
ROOT=/workspace/storage-shared/nlp/tungks/xtoken_all
SOURCE="$ROOT/ordures_ultra_v3"
PY=/opt/nemo_rl_venv/bin/python
RECIPE="$ROOT/data/math-code-20k-conversion.yaml"
if [ -f "$RECIPE" ] && [ -x "$PY" ]; then
  cd "$SOURCE" && "$PY" -B -m tools.xtoken_all.convert_math_code --config "$RECIPE"
else
  printf '%s\n' 'STOP: cần recipe đã inspect và interpreter có sẵn'
fi
echo "--- het Block M20K-02, shell con song ---"
)
```

Khi thành công, kiểm conversion.json: success=true, output_rows=20000,
task_counts={math:10000,code:10000}, source_order_preserved=true và SHA256.
Input thật chưa được convert/đọc trên máy phát triển; local test dùng fixture.

## Render và kiểm chứng tiếp

1. Dùng đúng launcher/render đã chạy thành công. Chỉ thay `--data` sang
   `/workspace/storage-shared/nlp/tungks/xtoken_all/data/math-code-20k-gym/train.jsonl`
   và tạo RUN_DIR mới. Giữ model, teacher, projection, runtime và Ray head của arm.
   Không launch vào node/GPU đang bận; không tự chuyển sang anhhbt.
2. Preflight kiểm toàn bộ prompt theo tokenizer/context đã chốt. Nếu dài quá,
   dừng để chốt budget hoặc tạo pool dẫn xuất có manifest riêng; không truncate
   prompt hay loại bài âm thầm trong converter để giữ claim 20k.
3. Smoke dùng mẫu có cả math và code, hai update và checkpoint như recipe đã
   kiểm chứng. Sample smoke là artifact riêng, không sửa pool train gốc.
4. Kiểm valid_samples/response_tokens bao gồm code; gym_code_qa_scores_excluded
   đếm code và gym_reward_sample_count chỉ đếm math hợp lệ. Batch toàn code có
   gym_reward_mean=null. Reward tuyệt đối không điều khiển mask hay update.
5. Dùng cùng output hash và quy tắc chọn prompt cho cả cross-tokenizer và
   same-tokenizer. Support null projection của helper same-tokenizer là việc
   riêng, gói này không tuyên bố đã hoàn tất nhánh đó hoặc P-KL/H-KL.

Không cần môi trường mới: dùng Python/PyYAML/PyArrow/tqdm đã có ở công ty.
Không đưa dataset, venv, model weights hoặc credentials vào ZIP bàn giao.
