# DSH — r16: Qwen3-4B gốc, non-thinking, cross Gemma / same Qwen32

## Nhánh tối thiểu hóa thay đổi code công ty

Nhánh `tung/algo/minimal-company` ghim `vllm_teacher@d92dc09`, giữ nguyên GRPO,
reward, async helper, OPD cũ và entrypoint GRPO theo mốc đó. X-Token dùng module
refit/data riêng; callback mới không bật cho caller GRPO cũ. Đọc
[company boundary](../../docs/guides/company-boundary.md) trước khi triển khai.
Đây là source mới, chưa được GPU canary sau khi tách; không dùng receipts của
pilot r15 hay canary Modal cũ để gọi nhánh này đã pass. Không sửa source/run đang
chạy, không tự rebuild runtime hoặc host lại teacher. Bản operator-reconcile và
evidence cũ giữ nguyên để đối chiếu.

## Đối chiếu báo cáo DSH ngày 2026-10-28: ba pilot r15 hoàn tất

Đây là snapshot từ transcript operator cung cấp, không phải lần Codex truy cập
node độc lập. Ba launch dưới đây có output `pilot-result.success:true`,
200/200 update, process/tee exit0, final checkpoint step200; listing có20 thư mục
step10..200. Chưa nhận bản file gốc để kiểm lại toàn bộ payload của20 checkpoint,
source hash/effective config từng run hoặc chất lượng held-out. Đây là campaign
recipe1.7B r15, tách biệt với10 canary Qwen3-4B/2GPU trên Modal ở nhánh gym-real.

| Arm | Run launch dưới `runs/` | Teacher / vLLM | Bằng chứng mode khi train |
| --- | --- | --- | --- |
| A cross | A-on-14-20261027-204156-3c2be9 | Gemma-4-26B-A4B-it /0.32.0 | deployment_config, verified:true |
| B same | B-on-02-20261028-110847-8e50d0 | Qwen3-32B /0.30.0 | upstream_default_assumption, accepted:true, verified:false |
| C same235 | C-on-01-20261028-131134-02361d | Qwen3-235B-A22B-Instruct /0.31.0 | upstream_default_assumption, accepted:true, verified:false |

Check-only attempts không có process-exit/pilot-result/checkpoints là đúng;
không tính chúng thành run fail. Giữ B-on-02-20261027-210946-e8cb03 exit1 làm
bằng chứng lỗi warmup eval-6. Không launch lại200 step chỉ vì có metadata mới.

### Quyết định về thay đổi operator

Merge sửa wrapper vào source dùng chung. Bản gốc `source ensure_runtime.sh --`
truyền literal `--` vào argparse. Chỉ bỏ `--` vẫn để Bash kế thừa pilot flags.
Bản vá dùng một function gọi không tham số, source setup trong cùng shell để
giữ exports, rồi chuyển nguyên pilot flags đến prepare_pilot. Đã kiểm actual
Bash với setup/interpreter stub: launch mặc định, check-only+GPU subset+argument
có dấu cách, và setup fail phải dừng trước pilot. Không giả lập GPU ở các test này.

Không ghi đè recipe gốc `pilot_200.yaml` hoặc `pilot_200_4b.yaml` bằng snapshot
node. Giữ thay đổi triển khai trong bản YAML operator của campaign và
`launch-input.yaml`/`effective.yaml` đã đóng trong từng run. Một spec shared bị
chỉnh sau khi A hoàn tất không chứng minh đó là input A đã dùng. Source r16 đã
có nghĩa là recipe4B, không gắn lại nhãn r16 cho campaign1.7B vừa hoàn tất.

Snapshot triển khai DSH báo: node cross -0-2 được thay pod thành
`rl-test-opd-ultra-selfbuilt-trainllm-1-0-3`; B/C chạy tuần tự ở -0-3.
Teacher A giữ `http://172.29.93.105:8000` cùng evidence gemma-032;
B chuyển `http://172.29.219.204:8001`, evidence null; C giữ
`http://172.29.131.162:8000`, evidence null. B/C null chỉ hợp lệ vì exact version
và measured scoring/parity được policy chấp nhận; không phải bỏ gate.
Node -0-3 có4 GPU UUID654c372c/8cb6790e/0df4fc7d/47300b26, Ray head PID875,
IP172.29.167.250 tại thời điểm probe. Đây là snapshot lịch sử, không phải
allocation/PID/IP để tái dùng trực tiếp cho lần launch mới. Muốn run4B tiếp thì
sửa bản deployment YAML4B riêng từ recipe4B và lấy allocation/evidence mới;
không copy file1.7B lên4B, không đổi run cũ, không resume chéo student.

QA venv copy và tar built-source18GB là thao tác runtime DSH báo đã làm, không
phải dependency/source được merge vào Git. Venv cần giữ prefix và qua actual
app/import/API probe; copy từ math_with_judge không tự chứng minh general_qa đúng.
Tar nguồn `_C` load được không tự chứng minh mọi native module hợp glibc.
DeepGEMM-off ở eval-6 chỉ qua CPU guard; campaign B/C cuối chuyển sang -0-3,
không dùng kết quả đó để tuyên bố eval-6 đã generation/train pass. Không rebuild
hay sửa runtime đang chạy, không dừng Ray/job của người khác.

### B/C có được eval ngay không?

Có. Held-out evaluation chấm weights student, không cần teacher đang chạy hoặc
mode teacher được nâng thành verified trước. Policy hiện hành chấp nhận exact
0.30.0/0.31.0 với nhãn assumption; giữ nhãn đó trong report. Muốn xác minh mode,
đọc args/config/image+deployment identity của server, không restart. Để gắn bằng
chứng vào run lịch sử, phải chứng minh deployment đó bao phủ thời gian training;
args của server hiện tại đơn độc không xác minh server quá khứ. Lưu supplemental
audit riêng, không sửa teacher-probe/effective YAML/receipt của run đã hoàn tất.
Nếu có bằng chứng mode thật khác raw_logprobs, đánh giá tác động lên training
objective trước khi dùng kết quả để kết luận về phương pháp.

Probe mass khoảng A0.9306/B0.0686/C0.7401 là một prompt probe, không phải retained
mass trung bình training và không phải điểm teacher. Không so train loss giữa
ba teacher thành quality ranking; support/top20 và tail có thể khác nhau.

### Bước tiếp: cùng một protocol held-out cho baseline+A+B+C

Không train thêm. Đầu tiên chốt file gốc của đúng ba run launch: pilot-plan,
provenance, launch-input, effective YAML, teacher-probe, process-exit, pilot-result
và metrics. Đối chiếu cùng initial student/tokenizer, data hash/order, seed,
non-thinking, context/response, batch/LR/schedule,200 updates, objective/top20.
Các khác biệt phải hiện trong report. Metadata lịch sử không được suy ra từ
YAML shared hiện tại; chưa đủ bằng chứng thì ghi unknown.

Dùng `export_checkpoint.py` hiện có (Automodel consolidate), `--inspect` trước,
`--validate-load` khi export, mỗi output thư mục mới. Giữ shards nguồn; không
đổi tên shard/đoán mapping, không serve export có FAILED. Xuất step200 của A/B/C;
baseline là đúng initial1.7B đã dùng, không phải initial4B hoặc A-on-11 weights.
Export được kiểm file/load không chứng minh optimizer resume parity.

Primary comparison: baseline cùng ba checkpoint step200; không chọn best20
checkpoint trên test set. Tái dùng launcher/scorer evalhub đã chạy; pin exact
dataset split/revision/hash, scorer version, chat template enable_thinking:false,
temperature/top-p, token budget, stop rules, sample count và seed giống nhau.
Loại/ghi nhận overlap với16367 prompt train trước khi gọi held-out. Cố định tập
sample/order; lưu responses và per-example scores để scorer có thể chạy lại mà
không regenerate. Missing/timeout không thành đáp án sai0 một cách âm thầm.
Report pass@1/accuracy, số bài thực chấm, truncation/failure, và paired differences
so với baseline; chi phí/tokens/wall-clock riêng. Một train seed/pilot chưa đủ
để kết luận phương pháp nào tốt hơn nói chung. Nếu xem learning curve, dùng cùng
mốc checkpoint cho tất cả arm và tách checkpoint selection khỏi final test.

Chưa có config/CLI evalhub thực trong transcript này nên chưa phát minh lệnh
eval hoặc launch job công ty. DSH gửi config và lệnh evalhub đã chạy cùng bốn
model/export paths để chốt lệnh dùng lại. Đây là bước còn thiếu cho quality;
không cản việc đóng training pilot thành mechanical pass có provenance caveat.

## Plan hiện hành r16; các mục r15 trở xuống để đối chiếu lịch sử

Gói r16 mặc định dùng tools/xtoken_all/pilot_200_4b.yaml, chỉ2 arm4B. File
pilot_200.yaml 1.7B giữ nguyên từ r15. Không chỉnh source/config/receipt/checkpoint
hoặc resume các run 1.7B cũ; giải nén source r16 vào thư mục mới dưới xtoken_all.
run_xtoken.sh mặc định chọn4B; không dùng wrapper pilot_200.sh cũ cho4B vì wrapper
đó vẫn đọc recipe1.7B. Cần recipe1.7B cho run mới thì đặt XTOKEN_PILOT_CONFIG
đến file pilot_200.yaml rõ ràng; không copy đè spec để đổi model.

Student là Qwen/Qwen3-4B bản gốc, cả weights/tokenizer tại
/workspace/storage-shared/models/Qwen3-4B. Đây là path dự kiến stage, chưa có
receipt assets4B trên node. DSH xác nhận đủ checkpoint và tokenizer; nếu path
khác thì sửa student.model_path và student.tokenizer_path trong YAML4B trước
render. Không dùng checkpoint Base/Instruct-2507/Thinking-2507 hoặc weights1.7B.

| Arm | Student node | Teacher | Prefix mới |
| --- | --- | --- | --- |
| cross | rl-test-opd-ultra-selfbuilt-trainllm-1-0-2 | Gemma-4-26B-A4B-it / http://172.29.93.105:8000 | A4-on-01 |
| same | anhhbt-eval-6-0-1 | Qwen3-32B / http://172.29.93.105:8001 | B4-on-01 |

DSH từ source/ trong gói mới: bash run_xtoken.sh cross --check-only trên node
cross; bash run_xtoken.sh same --check-only trên eval-6. Sau khi gate đạt, cùng
lệnh bỏ --check-only để train trong tmux. Suffix unique bảo vệ các run khác;
GPU/UUID/Ray phải kiểm live, không kill run1.7B để lấy4GPU. Nếu node còn bận thì
dừng/chờ hoặc operator đổi allocation; không tự chạy235B trong đợt4B này.

Giữ enable_thinking:false cho policy tokenizer và cả2 model HTTP Gym. Cùng data
16367 mẫu/hash của A-on-11,200 update,warmup20,cosine180,save10,keep20,
save_optimizer=true,is_async=false. Giữ context/response/batch/LR/refit4GiB
baseline. Hai arm độc lập cùng initial4B; baselineA-on-11 chỉ cấp asset/budget,
không cấp weights/optimizer. Chưa đo peak VRAM/full training4B trên node này.

Cross: YAML bật reuse_baseline_projection:true, projection_path:null. Launcher
so toàn bộ vocab token→ID và len của tokenizerA-on-11 với4B, giữ teacher tokenizer
cũ. Chỉ khi khớp mới dùng matrix baseline; pilot-plan.json ghi mapping hash và
SHA matrix đã dùng. Nếu mismatch, không skip gate: đặt reuse=false, cung cấp
projection_path mới đúng cặp tokenizer. Không ghi đè projection1.7B. Same giữ
projection=null; preflight kiểm tokenID/template/kwargs, probe server kiểm parity.

Teacher/evidence kế thừa r15 (vLLM0.32.0,raw_logprobs operator evidence tại
tmp/server-mode-gemma-032.json và tmp/server-mode-qwen32-032.json); probe mới vẫn
bắt buộc. Không restart/đổi teacher. ensure_runtime và script Super được tái dùng,
không rebuild/install. qualification.json ghi đúng test đã chạy; packet không
chứa weights4B hoặc runtime13GB, và chưa có kết quả GPU4B để gọi là pass pilot.

## Ghi nhận r15: sửa TeacherConfig và chuyển Qwen32 sang eval-6

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
