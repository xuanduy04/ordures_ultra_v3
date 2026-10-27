# DSH — ưu tiên chạy B same-tokenizer ngay, pilot 200 / warmup 20 / save 10

Không dùng pool math/code mới. Dùng đúng file toán A-on-11 đang train:
`/workspace/storage-shared/nlp/tungks/xtoken_all/data/dapo-r2-train/train.jsonl`
(16.367 dòng). Helper bắt path, row count và SHA256 trùng provenance A-on-11.
Nếu chưa thấy YAML/provenance A-on-11 trên node B, copy hai file đó sang cùng
path; không tạo provenance mới từ suy đoán và không copy checkpoint A sang B.

## Việc cần làm ngay

1. Giữ nguyên artifact/source A-on-11. Log mới cho thấy nó hoàn tất step66 rồi
   dừng ở step67 vì student decode có replacement character. Không sửa
   scheduler trong artifact cũ. Đã chạy warmup250 thì không thể đổi lịch sử
   thành warmup20. A-on-11 là baseline lịch cũ; preset chuẩn A-on-12 là run mới.
2. B chạy trên `anhhbt-eval-7-0-1`, 4 GPU 0–3. Teacher Qwen3-32B ở
   `http://172.29.191.69:8000`; xác nhận alias bằng probe /v1/models. Không host
   hoặc restart teacher. Không dùng GPU trên node A để chạy B.
3. Bung packet vào thư mục staging mới. Đặt `source/` thành checkout/overlay mới
   `/workspace/storage-shared/nlp/tungks/xtoken_all/ordures_ultra_v3_pilot200_r10`.
   Đây phải là source tree hoàn chỉnh từ packet; không đổi tên/ghi đè checkout A hoặc B đang chạy.
   Tái dùng 3rdparty/PYTHONPATH đã provision như r8, không cài dependency.
4. Source metadata pyproject/uv.lock/default_runtime_env/.python-version dùng
   `reference-base-metadata/` trong packet nếu thiếu. Không chạy install theo đó.
   Gym source và venv kế thừa từ A-on-11 đã restore; `prepare_pilot.py` có các
   CLI --gym-source/--gym-venvs/--gym-actor-python nếu node B dùng path khác.
5. Kiểm Ray head local thuộc job của bạn và chỉ có 4 GPU node B, theo R7b đã sửa
   Gym-workspace. Không dùng Ray head của A, không ray stop/kill process chưa rõ.
   Các block dưới nối head, không tự tạo hoặc restart Ray.

## Contract đã chốt

`pilot_200.yaml` là recipe theo run đang chạy. Nó chỉ đổi steps/scheduler/save,
teacher/tokenizer của B, Ray head và output paths. Giữ student checkpoint gốc,
learning-rate peak, batch4/micro1, seed, data shuffle/order, context, response,
refit4GiB, Gym sampling và objective top20+OTHER từ YAML A-on-11 thực tế.
Đây là ưu tiên nhất quán với run hiện tại, không tự ép context8k/response2k từ
profile train tổng quát. Hai profile train tổng quát cũng đã đổi 200/warmup20.

- 200 update mới, warmup20 với start_factor0.02, cosine180, giữ peak LR cũ.
- Save10,20,...,200; keep_top_k=20, optimizer included, synchronous finalize.
  Giữ đủ20 checkpoint định kỳ của pilot để so sánh các mốc. Không giảm retention
  hoặc xóa checkpoint để tiết kiệm disk nếu chưa được user yêu cầu.
  YAML mới chỉ có hiệu lực khi config được nạp; không tự đổi manager đang chạy.
- B projection_matrix_path=null và projection_sha256=null; không dummy identity.
Teacher tokenizer dùng `/workspace/storage-shared/models/Qwen3-32B`, kwargs giống student, head size
  kỳ vọng151936; teacher probe bắt /tokenize khớp ID trên ba mẫu và score top20.
  Local identity phải khớp vocab→ID, template/kwargs, context và padding grid.
- B dùng cùng loss coarse KL top20+OTHER với A, không đổi sang native OPD cũ,
  P-KL hoặc H-KL. Teacher Gemma/Qwen vẫn là biến khác nhau của hai recipe.
- A-on-11 lịch cũ không phải đối chứng scheduler hoàn toàn khớp với B mới.
  A-on-12 là preset cross chuẩn để chạy sau khi node A được giải phóng.
- Không auto-resume: mỗi run directory mới, initial weights từ path model gốc.

## B — render và kiểm tra

Block P200-B01 — chạy trên node anhhbt-eval-7-0-1 — render B từ pool/budget A-on-11 rồi kiểm runtime và teacher

```bash
(
export XTOKEN_ALL_ROOT=/workspace/storage-shared/nlp/tungks/xtoken_all
export NRL_PYTHON=/opt/nemo_rl_venv/bin/python
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export NRL_IGNORE_VERSION_MISMATCH=1
SOURCE="$XTOKEN_ALL_ROOT/ordures_ultra_v3_pilot200_r10"
export XTOKEN_RAY_ADDRESS="$(hostname -I | awk '{print $1}'):6379"
if [ "$(hostname -s)" = anhhbt-eval-7-0-1 ] && [ -f "$SOURCE/tools/xtoken_all/pilot_200.sh" ]; then
  cd "$SOURCE" && bash tools/xtoken_all/pilot_200.sh same render &&
    bash tools/xtoken_all/pilot_200.sh same check
  STATUS=$?
else
  printf '%s\n' 'STOP: sai node hoặc thiếu source pilot200'
  STATUS=1
fi
echo "P200_B01_EXIT_STATUS=$STATUS"
echo "--- het Block P200-B01, shell con song ---"
exit "$STATUS"
)
```

Ray address trên chỉ là địa chỉ local theo R7b port6379; phải xác nhận head
thuộc job trước launch. Nếu node dùng NIC/head khác, sửa biến ở block mới.

Teacher B trả /version `0.14.2.dev0+gd7de043d5.d20260219` và /v1/models alias
Qwen3-32B trong log R8, Ray B là172.29.167.243:6379. Policy không đoán default
cho version chưa review. Nếu probe báo
mode unconfirmed, giữ preflight thành công và cung cấp deployment evidence thật
theo schema probe cũ (`operator_checked_server_args` hoặc `server_reported_config`),
đặt `XTOKEN_TEACHER_MODE_EVIDENCE` tới file JSON rồi chạy lại action check.
Không sửa server, không tạo evidence giả, không sửa receipt thành success.
Nếu alias khác, dùng prepare_pilot.py --teacher-model <alias đã quan sát> với
run-id mới, thay vì sửa effective.yaml sau render. Run render lỗi chưa tạo run
cho phép chạy lại input plan nếu nội dung giống hệt; run đã tồn tại thì không ghi đè.

## B — allocation mới rồi launch

Sau khi check pass, DSH xác minh PID/cmdline Ray head là của job này và 4 GPU rảnh.
Chỉ sau xác minh ấy mới dùng flag operator-owned-ray-verified dưới đây.

Block P200-B02 — chạy trên node anhhbt-eval-7-0-1 — ghi allocation mới, kiểm stream và launch B trong foreground

```bash
(
export XTOKEN_ALL_ROOT=/workspace/storage-shared/nlp/tungks/xtoken_all
export NRL_PYTHON=/opt/nemo_rl_venv/bin/python NEMO_RL_VENV_DIR=/opt/ray_venvs
export NRL_IGNORE_VERSION_MISMATCH=1
SOURCE="$XTOKEN_ALL_ROOT/ordures_ultra_v3_pilot200_r10"
export PYTHONPATH="$SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export XTOKEN_RUN_DIR="$XTOKEN_ALL_ROOT/runs/B-on-01"
export XTOKEN_ALLOCATION_RECEIPT="$XTOKEN_RUN_DIR/allocation-$(date +%s).json"
export XTOKEN_RAY_ADDRESS="$(hostname -I | awk '{print $1}'):6379"
cd "$SOURCE" && (
  set -euo pipefail
  test "$(hostname -s)" = anhhbt-eval-7-0-1
  mapfile -t UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 4)
  "$NRL_PYTHON" -B tools/xtoken_all/check_allocation.py --node-label anhhbt-eval-7-0-1 \
    --expected-gpu-uuids "${UUIDS[@]}" --ray-address "$XTOKEN_RAY_ADDRESS" \
    --operator-owned-ray-verified --output "$XTOKEN_ALLOCATION_RECEIPT"
  bash tools/xtoken_all/pilot_200.sh same launch
)
STATUS=$?
echo "P200_B02_EXIT_STATUS=$STATUS"
echo "--- het Block P200-B02, shell con song ---"
exit "$STATUS"
)
```

Chạy B02 trong terminal/tmux do DSH quản lý, giữ process sống. Nếu receipt check
quá30phút hoặc allocation quá300giây, chạy lại đúng probe/allocation cần thiết,
không render lại khi effective config/source không đổi. Không chạy B02 hai lần.

Khi hoàn tất: từ source mới và cùng env, dùng
`bash tools/xtoken_all/pilot_200.sh same assess`. Checker mới đòi200 update,
policy versions0..199, scores top20, exit0 và finalized checkpoint200. Không dùng
check_result.py hai-step cho pilot200. Theo dõi driver.log ở run B-on-01;
report riêng quality eval held-out, chưa có eval tích hợp tự chạy.

## Cross chuẩn — chỉ sau khi node A rảnh

Từ source mới trên `rl-test-opd-ultra-selfbuilt-trainllm-1-0-1`, dùng cùng env
và Ray head local đã xác minh:
`bash tools/xtoken_all/pilot_200.sh cross render`, rồi `cross check`.
Nó tạo A-on-12, giữ teacher Gemma/projection/data/budgets của A-on-11, scheduler
200/20/save10. Allocation/stream/launch dùng chuỗi helper cũ với run A-on-12,
sau cùng `cross assess`. Không chạy cross launch khi A-on-11 còn giữ GPU.

## Phạm vi bằng chứng

Local chỉ có offline fixtures và Ray/tensor doubles; chưa chạy node B, chưa
probe teacher Qwen từ workstation, chưa có kết quả GPU cho bản này. Không cần
Modal, rebuild image hoặc tải model/data mới. Converter10k+10k vẫn trong source
nhưng không thuộc run này. Giữ nguyên server teacher và source job A-on-11.
