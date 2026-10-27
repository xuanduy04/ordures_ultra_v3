# DSH — r15: sửa TeacherConfig, cross Gemma và same Qwen32 trên eval-6

## Đọc phần này trước; các mục r14 trở xuống là lịch sử đối chiếu

R13/r14 có lỗi thực: factory Gym nhận TeacherConfig đã được Pydantic normalize
nhưng validator đọc teacher['remote']. Run A-on-13-20261027-200300-6dac53 dừng
ở đây sau load weights, chưa tạo Gym. R15 sửa cả raw dict/typed object, validate
sớm trước Policy/load weights và kiểm lại ở factory. Không bỏ kiểm URL/model QA.

Giải nén packet r15 vào thư mục mới dưới xtoken_all, vào source/. Không overlay
lên source job đang import, không sửa receipts/effective.yaml của run cũ. Root
metadata và ensure_runtime.sh đã kèm. Runtime archive13GB không nằm trong packet;
đã stage ở /workspace/storage-shared/nlp/tungks/xtoken_all/tmp/runtime-opt-20261027.tar.gz.
ensure_runtime.sh dùng path đó mặc định; healthy runtime bỏ qua giải nén sau khi
kiểm import/API thật. Không install/rebuild, không restart teacher hoặc kill Ray.

| Arm | Node chạy student | Teacher | Prefix mới |
| --- | --- | --- | --- |
| cross | rl-test-opd-ultra-selfbuilt-trainllm-1-0-2 | http://172.29.93.105:8000, gemma-4-26B-A4B-it | A-on-14 |
| same | anhhbt-eval-6-0-1 | http://172.29.93.105:8001, Qwen3-32B | B-on-02 |

YAML đã có4 UUID GPU đúng từng node. Dùng tools/xtoken_all/pilot_200.yaml trong
source r15, không chép nguyên spec r13 đè lại node/URL cũ. Shared evidence:
tmp/server-mode-gemma-032.json và tmp/server-mode-qwen32-032.json dưới xtoken_all,
do operator tạo từ args server. Teacher vLLM0.32.0 không được tự coi raw_logprobs
theo version; probe phải đọc evidence và chấm lại response/tokenizer. Không copy
receipt T12 cũ vào run mới. Nếu deployment khác evidence thì dừng và cập nhật
thông tin server thật; không sửa server tự động. Same235 chưa được chọn cho
eval-6, giữ recipe cũ để chạy sau khi có allocation/probe mới, không launch lúc này.

DSH trên node cross: từ source r15 chạy bash run_xtoken.sh cross --check-only.
DSH trên eval-6: từ source r15 chạy bash run_xtoken.sh same --check-only.
Check-only có thể start head nếu thiếu theo YAML, kiểm allocation và gọi scoring
teacher nhỏ; không train. Khi các gate đạt, dùng cùng lệnh bỏ --check-only trong
tmux trên node tương ứng. Mỗi lần tạo run ID unique; không resume A-on-13 fail.
Nếu GPU bận hoặc Ray khác allocation thì dừng, không tự kill process để lấy GPU.

Giữ student Qwen3-1.7B initial,16367 mẫu toán/hash A-on-11,200 update,warmup20,
cosine180,save10,keep20,save_optimizer=true; top20+OTHER,T1,CE0. Không gọi loss
xấp xỉ là P-KL/H-KL paper. Cross và same là hai run độc lập cùng initial weights.

Evidence operator T15/T17: eval-6 restore3 venv stores, torch2.10.0+cu129/cuda=True,
Ray2.55.1 head0/4GPU tại172.29.85.72. Đây là snapshot setup, vẫn kiểm lại allocation.
QA venv không có vLLM là bình thường: kiểm import app general_qa bằng helper cũ,
không pip install training stack vào service venv. Chưa có live pilot r15; xem
qualification.json/evidence cho test local và tái hiện typed-config thực tế.

## Ghi nhận r14

Thêm ensure_runtime.sh và test fixture, không thêm môi trường/dependency mới.
run_xtoken.sh gọi ensure trước bước setup Super. Cách chạy cross/same/same235 cũ
không đổi. Runtime đúng và import/API pass thì bỏ qua giải nén; không dựa marker.

Stage archive runtime đã audit một lần trên shared storage, mặc định:
xtoken_all/runtime/runtime-opt-20261027.tar.gz. Đổi XTOKEN_RUNTIME_ARCHIVE nếu khác.
Đặt XTOKEN_SUPER_SOURCE tới source nemo-rl đầy đủ đã bung từ custom_opt ZIP cũ;
đặt CUSTOM_NLTK_DATA_DIR tới nltk_data cạnh nó. Đây là source/assets có sẵn,
không cần lấy lại hoặc tải model teacher về node student. Chi tiết trong guide.

Một lệnh riêng: bash ensure_runtime.sh. Kiểm tra chỉ đọc: bash ensure_runtime.sh
--check-only. Không tự download/install hoặc ghi đè venv có sẵn. Nếu version,
source, ABI hay prefix không phù hợp thì dừng và giữ nguyên. Không kill/start Ray
hoặc sử dụng GPU để setup. Readiness không phải bằng chứng full training.

Kiểm chứng r14 là fixtures Linux nhỏ: cold install, reuse không cần archive,
hardlink, checksum, process guard, source binding và failed receipt. Không coi
fixtures là đã restore lại gói13GB trên image công ty hay đã test GPU.

## Ghi nhận r13

R13 chỉ sửa detector GCS trong helper cũ và regression test, không thêm file source.
Ray 2.55.1 nhận --port ở CLI nhưng truyền --gcs_server_port vào tiến trình con.
R12 chỉ tìm --port nên từ chối cả head có sẵn lẫn head tự start. Detector nay nhận
cả hai tên, dạng =value hoặc cặp tách, vẫn kiểm executable/UID/port chính xác;
port mâu thuẫn hoặc khác giá trị vẫn bị từ chối.

Đã test trên head Ray 2.55.1 thật trong Modal CPU: r12 từ chối cùng head; r13 auto
start thành công, reuse cả khi tắt/bật start, PID không đổi; ray.init và CPU task
thành công. Xem evidence/r13-real-ray-result.json trong packet. Không có GPU,
không gọi teacher hoặc node công ty; đây không phải kết quả full training.

Giữ nguyên head T9 tại172.29.167.231:6379 và run A-on-13 trên node
rl-test-opd-ultra-selfbuilt-trainllm-1-0-2. Không ray stop, không kill PID848,
không sửa effective.yaml/receipts cũ để giả là pass.
Giải nén source r13 vào thư mục mới dưới xtoken_all, không đè r12. R13 đóng gói
script shell bằng LF và kiểm bash -n từ chính payload; không cần tự convert .sh.
Nếu dùng patch thay full source, preimage manifest là Git blob LF; không áp mù
lên Python CRLF trích từ ZIP r12. Ưu tiên full source r13, giữ source cũ làm bằng chứng.

Trong source r13, giữ nguyên các giá trị node-specific đã xác minh trong YAML
(kể cả server_mode_evidence nếu có); không copy receipt pass từ run trước.
Chạy lại một lệnh check-only bên dưới. Detector sẽ reuse head T9 khi đúng UID/IP,
topology và GPU rảnh. Sau khi check pass, lệnh train tạo run ID mới và kiểm lại,
không ghi đè A-on-13. Schedule/student/data/loss không đổi so với r12.

## Launcher r12 được giữ nguyên trong r13

Launcher r12 thay quy trình nhiều block bên dưới. So với r11, source chỉ có
một file mới: `run_xtoken.sh`. Không thêm controller hoặc viết lại QA/converter.

Giải nén packet vào thư mục mới dưới `/workspace/storage-shared/nlp/tungks/xtoken_all/`,
vào `source/`. Metadata nguyên bản đã kèm; không copy đè source job đang chạy.
Chỉnh `tools/xtoken_all/pilot_200.yaml` rồi chạy trong tmux trên đúng student node:

| Một lệnh | Teacher | Student node |
| --- | --- | --- |
| `bash run_xtoken.sh cross` | Gemma-4-26B-A4B-it, `172.29.219.204:8001` | pod cross hiện tại `rl-test-opd-ultra-selfbuilt-trainllm-1-0-2` |
| `bash run_xtoken.sh same` | Qwen3-32B, `172.29.191.69:8000` | anhhbt-eval-7-0-1 |
| `bash run_xtoken.sh same235` | Qwen3-235B-A22B-Instruct, `172.29.131.162:8000` | anhhbt, sau khi same kết thúc và GPU rảnh |

Chỉ kiểm tra, chưa train: `bash run_xtoken.sh cross --check-only`. Check-only vẫn
có thể start head Ray local nếu thiếu, tạo run/receipt và gọi scoring teacher nhỏ.
Mỗi lần chạy sinh ID mới; không xóa plan/run cũ, không resume A-on-11 hoặc arm khác.

YAML chứa root, student, teacher, dataset, GPU và runtime. `student.model_path:null`
giữ initial checkpoint gốc trong A-on-11/effective.yaml, không phải checkpoint66.
Script chỉ giữ interpreter + path YAML và source đúng `use_custom_opt.sh` của Super.
Không install, rebuild, kill Ray, restart teacher hoặc tự bỏ gate.

Launcher dùng helper cũ theo thứ tự: allocation/owned local Ray → render →
preflight assets/import Gym → teacher probe → allocation mới → Ray stream probe →
launch.sh → check_pilot.py. Khóa từng GPU UUID để các launcher mới không đè nhau.
Ray cùng user root chưa chứng minh thuộc project nào: operator phải dành head/GPU
đó cho job này. Nếu đổi GPU subset trên head cũ, head phải có cùng CUDA visibility;
có thể đổi port tạo head riêng khi cần, không kill head của job khác.

Lỗi general_qa: teacher mới ở172.29.219.204:8001 nhưng QA vẫn hỏi172.29.85.69:8000.
Template A-on-11 đã resolve URL thành literal; helper cũ chỉ đổi teacher.remote.
Fork QA khởi tạo client và hỏi /v1/models kể cả should_use_judge=false. R12 cập nhật
URL/model QA trong renderer, kiểm lại ở preflight và factory trước tạo actor.
Không phải lỗi convert rl_qa/general_qa; không cần sửa server teacher.

Xem `docs/guides/xtoken-gym-super.md` để đổi student/tokenizer/projection hoặc
chọn raw dataset và gọi converter Super cũ từ chính YAML. Vẫn200 update, warmup20,
save10, keep20, top20+OTHER, T1/CE0. Cần metadata A-on-11 làm template asset/budget.
Không hứa mọi kiến trúc đều chạy được: vẫn phải qua runtime và tokenizer gates.

Gemma vLLM0.30.0 được chấp nhận theo upstream raw_logprobs default có nguồn docs;
receipt ghi assumption, không giả là đã kiểm args server. Dev/custom version như
Qwen32B vẫn cần evidence khi exact-version policy không nhận diện. Điền path vào
`arms.<arm>.server_mode_evidence`. Không tự đổi teacher để vượt gate.

Xem qualification.json cho số test local; chưa có live full pipeline của launcher
mới trên node công ty. Checkpoint A-on-11/step20 đã HF-load/finite-forward pass
trong log user trước đó, không convert lại.

## Ghi nhận r11 và wrapper từng bước (lưu để đối chiếu)

User đã chốt ba run độc lập; không multi-teacher trong một run:

| Arm CLI | Run mới | Teacher | Student node |
| --- | --- | --- | --- |
| cross | A-on-13 | Gemma-4-26B-A4B-it; r12 cập nhật URL trong YAML | rl-test-opd-ultra-selfbuilt-trainllm-1-0-2 |
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
