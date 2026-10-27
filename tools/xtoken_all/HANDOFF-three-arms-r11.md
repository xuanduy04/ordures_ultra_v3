# DSH — r11: ba run độc lập, cùng Qwen3-1.7B, 200 step, giữ đủ20 checkpoint

User đã chốt ba run độc lập; không multi-teacher trong một run:

| Arm CLI | Run mới | Teacher | Student node |
| --- | --- | --- | --- |
| cross | A-on-13 | Gemma-4-26B-A4B-it từ YAML A-on-11 | rl-test-opd-ultra-selfbuilt-trainllm-1-0-1 |
| same | B-on-02 | Qwen3-32B ở172.29.191.69:8000 | anhhbt-eval-7-0-1 |
| same235 | C-on-01 | Qwen3-235B-A22B-Instruct ở172.29.131.162:8000 | anhhbt-eval-7-0-1, sau B |

32B là alias teacher đã quan sát, user gọi gọn30B; không tự đổi sang Qwen3-30B-A3B.
Endpoint235 là thông tin user cung cấp từ trước, CHƯA probe lại trong đợt này.
Nếu alias/URL hiện tại khác, cập nhật recipe trong source mới trước render hoặc
dùng prepare_pilot.py --teacher-url/--teacher-model đã đọc từ server, rồi run ID mới.
Không tự host/restart/sửa bất kỳ teacher nào.

## Chung cho cả ba

- Initial student weights giống nhau: path gốc trong effective.yaml A-on-11.
  Không lấy checkpoint student đã distill làm init cho arm kế tiếp.
- Đúng file toán16367 dòng của A-on-11, cùng SHA, seed, thứ tự/shuffle, batch,
  context, response cap và peak LR. Helper giữ budget thực từ A-on-11.
- 200 update, warmup20, cosine180, save mỗi10, keep_top_k20, save_optimizer=true.
  Giữ đủ10,20,...,200; không tự giảm retention/dọn checkpoint.
- Cùng top20+OTHER, T1, CE0. Hai arm Qwen projection=null; cross dùng projection
  Qwen→Gemma đã review. Không đổi loss, không gọi đây là P-KL/H-KL nguyên bản.
- Mỗi run directory mới, không resume optimizer/dataloader/policy từ A-on-11
  hoặc arm khác. Giữ A-on-11 làm artifact lịch cũ, không ép lịch sử warmup250
  thành warmup20. Không dùng lại run IDs r10 có thể đã render/start.

## Nguồn và phân bổ

Giải nén source đầy đủ từ packet vào checkout/overlay mới dưới
`/workspace/storage-shared/nlp/tungks/xtoken_all/ordures_ultra_v3_pilot200_r11`.
Copy metadata tham chiếu nếu thiếu như bàn giao r10. Giữ original r10/r8 và
mọi source đang được job import. Tái dùng runtime Super/Gym/venv đã provision.

Hai student node đều là4GPU theo allocation đã chốt. A và B có thể chạy song
song khi đúng GPU rảnh; C xếp sau B trên cùng nodeanhhbt. Wrapper không có
scheduler tự chờ: KHÔNG gọi launch C khi B còn chạy hoặc GPU còn process.
Nếu muốn ba run song song cần node/allocation student thứ ba, chưa được chỉ định.

## Teacher/tokenizer gates

B dùng tokenizer Qwen3-32B tại /workspace/storage-shared/models/Qwen3-32B.
C mặc định dùng tokenizer student Qwen3 đã stage để chấm cùng sampled IDs,
không tải weights235B về node student. Đây là giả định shared tokenizer phải
kiểm bằng probe /tokenize + score; nếu sai, dừng và cung cấp đúng tokenizer235B
qua --teacher-tokenizer hoặc source recipe trước render. Không tự fix vocab IDs.

Mọi run phải qua local tokenizer identity/projection, teacher top20, runtime/Gym,
Ray và allocation riêng. B có version0.14.2.dev0+gd7de043d5.d20260219 quan sát
trong log, mode evidence vẫn cần nếu chưa được xác nhận. C trước đây được báo
0.31.0/top20; phải probe lại, không copy receipt cũ hoặc mặc định server còn sống.
Mỗi helper check ghi receipt theo đúng effective.yaml/hash; không bỏ gate vì gấp.

## Lệnh cùng wrapper cũ, thêm arm same235

Trên đúng node, đặt XTOKEN_ALL_ROOT, NRL_PYTHON=/opt/nemo_rl_venv/bin/python,
NEMO_RL_VENV_DIR=/opt/ray_venvs, NRL_IGNORE_VERSION_MISMATCH=1 và
XTOKEN_RAY_ADDRESS tới head LOCAL thuộc job đã kiểm tra. Từ source mới:

1. `bash tools/xtoken_all/pilot_200.sh <arm> render`
2. `bash tools/xtoken_all/pilot_200.sh <arm> check`
3. Tạo allocation receipt mới bằng check_allocation.py, đúng4 GPU UUID và Ray
   ownership, rồi export XTOKEN_ALLOCATION_RECEIPT tới receipt ấy.
4. `bash tools/xtoken_all/pilot_200.sh <arm> launch` trong tmux/terminal quản lý.
5. Khi kết thúc, `bash tools/xtoken_all/pilot_200.sh <arm> assess`.

Chỉ dùng arm cross/same/same235 từ bảng. Run ID lấy từ pilot_200.yaml,
wrapper tự chọn A-on-13/B-on-02/C-on-01. Nếu override run ID qua prepare helper,
phải dùng direct helpers + XTOKEN_RUN_DIR đúng path, tránh wrapper trỏ mặc định.

## Trạng thái sẵn sàng

Source/config/helper đã chuẩn bị cho cả ba. Chưa có live teacher/runtime/GPU
receipts cho các run mới: không gọi toàn bộ ba run là đã qua gate hoặc đã launch.
DSH chạy check trên hai node; probe235B là phần mới cần đóng trước C.
Quality eval held-out vẫn là bước riêng, cùng protocol cho cả ba; giảm KD loss
không tự chứng minh chất lượng math tăng. Checkpoint100/200 đều được giữ để eval.

R11 cũng sửa retention3→20. Job đã chạy giữ retention trong RAM nên sửa YAML
không cập nhật job ấy; xem HANDOFF-retention-r11.md. Export A-on-11/step20 đã
HF-load/finite-forward pass trong log user, không convert lại chỉ vì đổi plan.
