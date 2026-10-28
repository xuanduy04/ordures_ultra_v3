# X-Token qua NeMo Gym trên fork Super

Contract G1.1: Qwen3-1.7B sinh một response qua NeMo Gym `general_qa`, teacher
Gemma-4-26B-A4B-it chấm response bằng tokenizer riêng. Objective vẫn là
`topk_tail_kl`, K=20 tại mỗi vị trí + OTHER, CE=0, T=1. Đây là adaptation;
không gọi nó là P-KL nguyên bản, PPO hay sampled-token MOPD của GRPO.

## Bản r16: student Qwen3-4B gốc, tắt thinking

Cập nhật operator2026-10-28: transcript DSH đã báo ba pilot recipe1.7B r15
A-on-14/B-on-02/C-on-01 hoàn tất200/200, exit0, step200 và20 checkpoint directories.
Đọc phần đối chiếu mới ở đầu
[handoff](../../tools/xtoken_all/HANDOFF-three-arms-r11.md) trước snapshot node
bên dưới. A có deployment-mode evidence; B/C giữ nhãn accepted assumption.
Đây chưa phải quality evaluation và không phải full pilot4B. Recipe gốc được
giữ nguyên; thay node/URL/evidence nằm trong deployment config và receipt từng run.

Wrapper đã sửa boundary tham số setup: dùng function không tham số để source
`ensure_runtime.sh`, giữ exports trong shell gọi, không chuyển literal `--` hoặc
các flag pilot vào argparse setup. `--check-only` của launcher vẫn có thể setup
runtime/Ray trước preflight; muốn chỉ đọc runtime dùng helper trực tiếp với
`bash ensure_runtime.sh --check-only`. Regression chạy actual Bash gồm default,
check-only/GPU subset và setup-failure; không thay acceptance gate hoặc runtime.

Launcher `run_xtoken.sh` nay mặc định đọc `tools/xtoken_all/pilot_200_4b.yaml`.
Recipe 1.7B `pilot_200.yaml` giữ nguyên từng byte từ r15; source, run directories,
effective config, optimizer và checkpoint 1.7B cũ không được sửa hoặc resume.
Muốn tạo run 1.7B mới thì chọn file cũ bằng `XTOKEN_PILOT_CONFIG`; wrapper từng
bước `pilot_200.sh` cũng vẫn thuộc recipe 1.7B cũ, không dùng nó cho run 4B.

Student weights và tokenizer đều trỏ `/workspace/storage-shared/models/Qwen3-4B`,
là checkpoint chính thức `Qwen/Qwen3-4B`, không phải Base/Instruct-2507/Thinking-2507.
Đây là staging target dự kiến, chưa được kiểm tra tồn tại trên node. DSH xác nhận
đủ weights/tokenizer trước launch; nếu stage ở nơi khác thì chỉnh cả hai path
trong YAML. Giữ `enable_thinking:false` ở tokenizer policy và cả hai model Gym.

| Arm 4B | Run prefix unique | Student node | Teacher |
| --- | --- | --- | --- |
| cross | A4-on-01 | rl-test-opd-ultra-selfbuilt-trainllm-1-0-2 | Gemma-4-26B-A4B-it tại172.29.93.105:8000 |
| same | B4-on-01 | anhhbt-eval-6-0-1 | Qwen3-32B tại172.29.93.105:8001 |

Mỗi lần launcher sinh suffix mới. Giữ nguyên data/hash16367 mẫu của A-on-11,
context/response/batch/LR baseline;200 update,warmup20,cosine180,save10,keep20,
save_optimizer=true. Cả hai arm khởi tạo từ cùng initial Qwen3-4B, không từ
checkpoint 1.7B hay weights đã distill ở arm khác. Refit buffer vẫn4GiB như r15.
Chưa có phép đo VRAM/full pilot4B; không tự tăng context hoặc batch vì đổi model.

Cross bật `reuse_baseline_projection:true` với `projection_path:null`: launcher
load hai tokenizer local baseline/4B và so toàn bộ token→ID cùng len. Chỉ khi
giống hệt và teacher tokenizer giữ nguyên mới dùng path projection A-on-11;
ghi mapping hash, path và SHA matrix vào `pilot-plan.json.projection_reuse`.
Nếu khác, dừng; chọn `reuse_baseline_projection:false` và điền projection_path
đã tạo/kiểm cho cặp tokenizer mới. Không sửa matrix cũ. Đây là kiểm tương thích
mapping, không phải bằng chứng chất lượng distillation.

Same giữ projection=null, kiểm vocab/template/kwargs với tokenizer Qwen32 local
và parity/scoring trên server như r15. Chưa thêm arm 4B→235B. Node/UUID/evidence
được kế thừa từ snapshot r15; mỗi run phải qua allocation/Ray/teacher gate mới.
Không chiếm GPU của run 1.7B đang chạy; chờ GPU rảnh hoặc operator cấp node khác.

Từ source r16 mới trên đúng node: `bash run_xtoken.sh cross --check-only` hoặc
`bash run_xtoken.sh same --check-only`, rồi bỏ `--check-only` để train sau khi
các gate đạt. Không chép spec 1.7B đè lên file4B. Các mục r15 trở xuống mô tả
lịch sử; dùng bảng r16 và handoff r16 ở đầu file HANDOFF-three-arms-r11.md.

## Bản r15: typed config và phân bổ mới

Lỗi A-on-13-20261027-200300-6dac53 xảy ra trước khi tạo actor Gym:
`TeacherConfig` là Pydantic model nhưng validator r13/r14 đọc bằng `['remote']`.
R15 đọc trường bắt buộc qua mapping hoặc attribute, giữ kiểm tra URL/model QA.
Cùng validator chạy khi validate on-policy trước khi tạo Policy/load weights,
và chạy lại ngay trước tạo actor. Không bỏ gate hoặc chuyển config sai thành pass.
Regression dùng đúng định nghĩa `TeacherConfig` từ source, cả input dict, typed
teacher, factory tạo actor và stale endpoint; không chỉ test renderer với dict.

`tools/xtoken_all/pilot_200.yaml` là cấu hình chạy hiện tại:

| Arm | Student node | Teacher | Run prefix |
| --- | --- | --- | --- |
| cross | rl-test-opd-ultra-selfbuilt-trainllm-1-0-2 | http://172.29.93.105:8000 / gemma-4-26B-A4B-it | A-on-14 |
| same | anhhbt-eval-6-0-1 | http://172.29.93.105:8001 / Qwen3-32B | B-on-02 |

Hai server được operator báo vLLM0.32.0; YAML trỏ tới evidence đã tạo trên shared
storage: `tmp/server-mode-gemma-032.json` và `tmp/server-mode-qwen32-032.json`.
Không suy raw_logprobs từ version0.32.0; probe vẫn kiểm nội dung evidence,
tokenizer và top20 scoring. Mỗi run cần receipt mới theo effective.yaml mới.
Same dùng đúng4 GPU UUID của eval-6 đã gửi trong T14; trạng thái GPU/Ray phải
kiểm lại lúc chạy. Same235 giữ recipe cũ chưa reassigned/probe; không tự launch.

Thực thi từ source mới, không vá source đang được process khác import:
`bash run_xtoken.sh cross --check-only` hoặc `bash run_xtoken.sh same --check-only`.
Khi kiểm tra đạt, bỏ `--check-only` trên đúng node để tạo run mới và train.
Giữ200 step, warmup20, save10, keep20, cùng initial Qwen3-1.7B và data A-on-11.
Không resume run fail hoặc chuyển checkpoint cross sang same.

T15/T17 trong log operator xác nhận eval-6 đã restore /opt, torch.cuda=True,
Ray2.55.1. Đây là bằng chứng setup từ node, chưa chứng minh full pilot r15.
QA service venv không cần import vLLM; dùng probe app general_qa hiện có,
không cài vLLM vào thin service để chữa một lỗi import không thuộc contract.

## Setup runtime trên node mới (r14)

`bash ensure_runtime.sh` kiểm tra runtime thực tế; runtime phù hợp thì không đọc
hay giải nén archive. Không dùng file marker để quyết định bỏ qua. `run_xtoken.sh`
gọi bước này tự động trước setup Super và các gate chạy train cũ.

Chỉnh các path ở đầu ensure_runtime.sh hoặc dùng biến môi trường tương ứng:

| Biến | Nội dung |
| --- | --- |
| XTOKEN_RUNTIME_ARCHIVE | Archive đã stage trên shared storage; mặc định xtoken_all/tmp/runtime-opt-20261027.tar.gz theo T15 |
| XTOKEN_SUPER_SOURCE | Source nemo-rl đầy đủ trong ZIP Super cũ, có3rdparty; mặc định đường dẫn shared custom_opt_ultra_selfbuilt_nano/nemo-rl đã dùng |
| CUSTOM_NLTK_DATA_DIR | NLTK punkt_tab từ ZIP cũ; mặc định thư mục nltk_data cạnh Super source |

Archive được nhận bằng size/SHA256 cố định của bản đã audit, không theo filename.
File trên shared storage có thể đổi tên; không tự tải lại trên từng node, không
install pip/uv/apt, không sửa driver/image. Bootstrap chỉ cần Linux x86_64,
Python>=3.9 chuẩn và GNU tar; không cần PyYAML ở Python bootstrap.

Khi thiếu môi trường, script kiểm tra các phần hiện có trước, kiểm SHA archive,
giải nén vào staging riêng (cần45GiB trống), rồi chỉ publish các venv còn thiếu.
Giữ Python tại prefix /root gốc và các venv ở /opt. Bảo toàn hardlinks. Không merge
vào venv có sẵn, không ghi đè môi trường khác phiên bản. Phần đã tồn tại nhưng
không phù hợp sẽ dừng và giữ nguyên; helper không tự sửa ABI hay libc của image.

Nếu /opt/nemo-rl chưa có, helper có thể tạo symlink tới đúng Super source trên
shared storage. Binding đã tồn tại được giữ nguyên, kiểm chứng cuối dùng import/API
thật; không tự thay source của image. Gym source phải qua contract đã review,
và NLTK English punkt_tab phải có. Nếu đổi XTOKEN_SUPER_SOURCE, cũng đặt
runtime.gym_source trong pilot_200.yaml tới source Gym tương ứng.

Script không kill/start Ray, không chọn GPU, không load model. Trước khi publish
prefix hay tạo binding, kiểm process đang dùng runtime; giữ nguyên nếu có process.
Readiness gồm driver version/import,3 worker API và3 app Gym, không khởi động
HTTP service. Receipt nằm /tmp theo UID/node, không đặt marker chung trên shared
storage; mọi lần đều probe lại, không copy trạng thái ready giữa hai container.

`bash ensure_runtime.sh --check-only` là kiểm tra chỉ đọc, không tạo source alias,
không restore, không ghi receipt. Khác với run_xtoken.sh cross --check-only:
lệnh launcher vẫn có thể setup runtime/Ray trước khi dừng ở preflight.

READY ở đây là đủ để vào preflight; không chứng minh GPU, NCCL, refit, full train
hay Ubuntu22/24 đều tương thích. Image mới vẫn phải qua các kiểm chứng tương ứng.

## Sửa nhận diện Ray và đóng gói Linux (r13)

GCS của Ray2.55.1 dùng --gcs_server_port, dù CLI ray start nhận --port. R13 nhận
cả hai tên ở dạng =value hoặc cặp tách; vẫn kiểm cùng UID, executable, port và
topology. Không dùng socket mở làm bằng chứng ownership. Regression trên head
Ray2.55.1 thật ở Modal CPU đã tái hiện r12 từ chối, r13 auto-start/reuse đúng cùng
PID và chạy CPU task. Không thay thế kiểm tra nvidia-smi/allocation hay train GPU.

Packet r13 xuất source với core.autocrlf=false, kiểm script shell LF và bash -n
sau trích xuất. ZIP r12 từng bị Git archive chuyển sang CRLF dù blob trong commit
là LF. Dùng full source r13 ở thư mục mới, không convert mù source đang chạy.
Head T9 của công ty được giữ nguyên; check-only mới phải reuse, không yêu cầu
ray stop/restart để vượt detector.

## Entry point đơn giản (r12, được giữ trong r13)

Từ source mới của gói, chạy `bash run_xtoken.sh cross`, `bash run_xtoken.sh same`
hoặc `bash run_xtoken.sh same235` trên đúng student node, trong tmux.
`bash run_xtoken.sh cross --check-only` kiểm setup, teacher và Ray rồi dừng.
Chỉ có một file source mới là launcher shell; nó gọi setup Super-120B
`custom_utils/utils_training/use_custom_opt.sh` và các helper cũ, không thay
converter, Gym server hay vòng training.

Chỉnh `tools/xtoken_all/pilot_200.yaml` trước chạy:

| Mục | Ý nghĩa |
| --- | --- |
| `root`, `baseline_run` | Shared root và run template; chỉ lấy initial assets/budgets, không resume checkpoint |
| `student.model_path`, `tokenizer_path` | null giữ original; thay model thì tokenizer mặc định theo model mới |
| `student.chat_template_kwargs` | Áp vào tokenizer và cả hai Gym model proxy |
| `arms.<arm>.teacher_url`, `teacher_model`, `teacher_vocab_size` | Base URL không có /v1, alias /v1/models, kích thước head teacher |
| `teacher_tokenizer`, `projection_path` | Asset local phù hợp; cross null kế thừa map cũ, same luôn projection=null |
| `data_path`, `expected_rows`, `data_sha256` | Dataset chọn rõ; đổi bộ thì đổi SHA/count hoặc đặt null để ghi nhận mới |
| `dataset_conversion` | null cho general_qa JSONL đã chuẩn hóa; raw dùng converter cũ theo ví dụ dưới |
| `runtime` | GPU indices, existing worker/Gym venv, source Gym; ray_host=null lấy IP pod hiện tại |
| `arms.<arm>.gpu_uuids` | UUID allocation đã chốt; null chụp GPU local được chọn, vẫn kiểm occupancy/Ray |
| `arms.<arm>.server_mode_evidence` | Path JSON evidence đã kiểm args server; null dùng exact-version policy khi được phép |

Chuyển raw math bằng converter Super hiện có: thay data_path sang raw, khai đúng
SHA/count và đặt ngay trong YAML này:

```yaml
dataset_conversion:
  kind: math
  options:
    prompt_field: prompt
    answer_field: reward_model.ground_truth
    dataset: selected_math
```

Đổi field theo schema thực, không đoán schema theo ví dụ. Math yêu cầu expected_rows
dương. Pool math_code dùng kind:math_code và mapping/count trong recipe hiện có
`tools/xtoken_all/math_code_20k.yaml` làm options (bỏ input/output_dir). Cả hai gọi
converter cũ, giữ mẫu/thứ tự, không resample. Cache theo hash input/recipe, xác minh
receipt/hash trước reuse. Output nằm root/data; raw có thể được đọc ở path khác.
QA reward của code không phải execution pass.

Đổi tokenizer cross cần projection_path tương ứng; launcher từ chối kế thừa map
khi thấy path tokenizer đổi. Shape gate không chứng minh map đúng ngữ nghĩa: phải
build/review map với đúng hai tokenizer. Same phải qua identity + teacher probe.
Hỗ trợ kiến trúc khác còn phụ thuộc runtime; launcher không tự cài dependency.

Mỗi lần tạo run ID mới. Ray chỉ tự start local head khi cổng rảnh, reuse head thuộc
user trên pod hiện tại, không stop/kill job. Operator phải dành head/GPU đó cho job;
không dùng head chung project khác. Khi đổi GPU subset, head phải có cùng CUDA
visibility; có thể đổi port để tạo head riêng, không kill head cũ. Cross và same ở
hai node đã chỉ định; same235 chờ same kết thúc trên node anhhbt.

Lỗi general_qa trong log mới là URL judge literal cũ còn sót sau khi teacher đổi
IP, không phải lỗi convert dữ liệu. Renderer nay cập nhật QA URL/model từ teacher;
preflight và runtime factory từ chối stale config trước spinup actor. Fork QA vẫn
hỏi /v1/models khi khởi tạo dù tắt judge inference. Không cần restart teacher để
sửa địa chỉ trong client YAML.

vLLM0.30.0: default raw_logprobs đối chiếu [docs chính thức](https://docs.vllm.ai/en/v0.30.0/cli/serve/#--logprobs-mode).
Receipt phân biệt upstream default assumption với bằng chứng deployment thực.
Dev/custom version không tự được suy về release gần nhất.

## Nguồn và thư mục

Mọi source mới, data dẫn xuất, assets và run output thuộc
`/workspace/storage-shared/nlp/tungks/xtoken_all/`. Checkout mới dùng
`xtoken_all/ordures_ultra_v3/`; giữ path cũ nếu đã có run, không tự di chuyển.
Không ghi vào checkout `/workspace/storage-shared/nlp/vai-1.4-30b/nemo-rl/`
hoặc `/opt/nemo-rl`; tái dùng interpreter, venv và dependency đã provision.
Không sửa/restart server teacher.

Nhánh `tungks/feat/xtoken-all-super` kế thừa X-Token `70534ca` và delta từ
bốn tarball công ty ngày 2026-10-26. Xem
`tools/xtoken_all/company-artifacts.json` để đối chiếu nguồn gốc từng gói.
Các delta Gym/GRPO/penalty và test của công ty được giữ cùng adapter mới.
Delta `custom_utils` được giữ nguyên theo gói, không refactor converter gốc.

G1.1 đã đọc thêm `custom_opt_ultra_selfbuilt.zip` và Code A/B Super. Root metadata
khớp; 142 file Python NeMo chung của ZIP và Code B giống nhau. Các snapshot này
không thay delta đã merge từ tarball 1711. `super-source-contract.json` chốt hash
60 file Gym cần cho luồng hiện tại, cùng API FSDP/checkpoint đọc từ archive.
`audit_archives.py` tái tạo contract và gói `reviewed-gym-source.zip` khoảng 174 KB;
đây là source, không kèm venv/native library và không phải lệnh cài dependency.

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

Pool hỗn hợp nay có adapter KD riêng: `tools/xtoken_all/convert_math_code.py`,
recipe `tools/xtoken_all/math_code_20k.yaml`. Chạy bằng module từ source root,
`--config <recipe> --inspect` trước, bỏ `--inspect` khi schema đã khớp. Recipe
khai chính xác 10.000 math + 10.000 code; mapping ability/data_source phải được
review theo output inspect, không suy task từ nội dung prompt hoặc lấy mẫu lại.
Converter QA cũ vẫn từ chối code để giữ contract của các run DAPO hiện tại.

Adapter hỗn hợp giữ nguyên thứ tự, prompt, nhãn/test case gốc trong metadata và
SHA256. Label code có cấu trúc được JSON-serialize vào expected_answer ngoài input
sinh; general_qa vẫn có thể tính điểm text nhưng đó không phải execution reward.
`xtoken_data` đánh dấu code không có QA metric hợp lệ; runtime loại điểm ấy khỏi
`gym_reward_mean` và ghi sample count/excluded count. KD mask không phụ thuộc
reward, không bỏ bài code và không dùng reward trong loss. Đây là data cho KD,
không phải môi trường RL code đã có verifier hay phép đo pass@1.

Handoff riêng tại `tools/xtoken_all/HANDOFF-math-code.md`; chuyển file thật trên
node, không tải pool công ty về workstation. Việc đổi dataset cần run directory
mới và render/preflight lại, không thay data của job đang chạy hoặc resume
dataloader từ pool DAPO khác. Chưa kiểm chứng file thật 20k trên node.

Smoke 128 prompt tokens + 32 response tokens không phù hợp mọi mẫu DAPO.
`make_smoke.py` tạo một bộ 8 mẫu đầu tiên thỏa budget, giữ thứ tự, ghi chỉ số
dòng nguồn/token count và hash; không sửa pool train. Không coi smoke này là
tập đánh giá chất lượng. Sau smoke phải chốt budget và dataset pilot riêng.

## Cấu hình và runtime Gym

Điểm vào: `examples.run_xtoken_on_policy_distillation` với config được render
từ `examples/configs/xtoken_gym_qwen3_gemma4.yaml`. Renderer bắt buộc source,
data, projection và run directory nằm dưới `XTOKEN_ALL_ROOT`, chọn đúng kiến trúc
Qwen3-1.7B. Smoke: 4 GPU 0–3, TP=CP=1, GBS=4/MBS=1, 2 updates, checkpoint bật.
Những số này không phải cấu hình mặc định cho full training.

r8 đặt refit buffer của recipe Qwen→Gemma/Gym thành 4 GiB tổng, hai vùng 2 GiB.
`XTOKEN_REFIT_REQUEST` ghi knob resolve và tổng byte ở driver trước refit;
`NRL_REFIT_BUFFER` ghi tổng byte, capacity và allocation sau alignment cho mỗi
vùng ở từng policy worker, trước cấp phát CUDA. Đây là bộ đệm chuyển trọng số,
không phải projection; context/batch không đổi kích thước tensor embedding.

Smoke `A-on-10` lưu sau update1 và update2 dưới `${XTOKEN_RUN_DIR}/checkpoints/`,
save_period=1, keep_top_k=2, save_optimizer=true, safetensors, không consolidated,
is_async=false. Dùng nguyên save/finalize của training loop và Automodel, không
thêm đường serialize riêng. Policy probe kiểm signature save_model/save_optimizer.
`check_result.py` đòi finalized step_1 và step_2, training_info/config/dataloader,
payload weights/optimizer/tokenizer có file không rỗng. Đây là file-closure check;
không chứng minh load/resume parity, không deserialize pickle trong checker.

YAML train riêng `examples/configs/xtoken_gym_qwen3_gemma4_train.yaml` kế thừa cùng
pipeline: student context8192, prompt6144, response2048, teacher budget16384,
GBS4/MBS1, tối đa200 update/1epoch, warmup20 rồi cosine180,
lưu mỗi10 step và step cuối, giữ20 checkpoint theo manager Super. Đây là profile
đề xuất có thể sửa trước launch, chưa phải cấu hình tái lập paper. Render bằng
`render.py --config <đường dẫn YAML train> ...` cùng các path đã có, nhưng data
phải là pool train đã convert/kiểm budget; không dùng smoke8 cho200 update.
Preflight từ chối khi rows/epochs không đủ requested steps hoặc prompt vượt budget.
Smoke256 vẫn là profile mặc định; renderer không tự chuyển sang train8k.

Pilot khẩn cấp lấy budget thực từ A-on-11 qua `prepare_pilot.py`, recipe
`pilot_200.yaml`, thay vì tự ép profile8k. Dùng nguyên pool toán16.367 dòng,
kiểm path/row count/hash với provenance A-on-11, không đổi sang math/code20k.
B-on-02 chạy same-tokenizer Qwen3-32B ở node anhhbt-eval-6-0-1 theo phân bổ r15;
A-on-14 là prefix cross mới sau lỗi A-on-13, xem bảng hiện hành ở đầu guide.
Giữ nguyên artifact A-on-11; lịch warmup250 đã thực thi không thể trở thành
warmup20 hồi tố.

Renderer nay nhận một trong `--projection <path>` hoặc `--same-tokenizer`;
same-tokenizer cần `--teacher-vocab-size` từ metadata model. Preflight kiểm
identity token→ID/template/kwargs/context/padding; provenance/launcher nhận
projection path và hash null, không tạo matrix identity. Profile same-tokenizer
riêng ở `xtoken_gym_qwen3_same_tokenizer_train.yaml`. Teacher tokenizer Qwen có
thể dùng bản student đã stage; probe teacher vẫn phải kiểm /tokenize và top20
thực. Không được coi identity local là bằng chứng tokenizer server đã khớp.

`tools/xtoken_all/HANDOFF-pilot-200.md` chứa lệnh render/check/launch B, giữ nguyên
runtime Super và teacher. `pilot_200.sh` gọi các helper đã có; `check_pilot.py`
kiểm đủ200 update và checkpoint cuối, không dùng checker smoke hai-step.
Save mỗi10, giữ20 checkpoint: đủ các mốc10..200 của pilot. Không tự giảm số
checkpoint hoặc dọn để tiết kiệm disk. Run đang chạy không tự đổi retention khi
sửa YAML trên đĩa; checkpoint đã xóa không được phục hồi bằng đổi config.
Chưa bật held-out evaluation tự động trong Gym.

R11 thêm arm `same235`/C-on-01, độc lập với cross/A-on-13 và same/B-on-02:
teacher Qwen3-235B-A22B-Instruct, cùng student init Qwen3-1.7B và toàn bộ data,
budget, loss top20+OTHER. Profile riêng
`xtoken_gym_qwen3_235b_same_tokenizer_train.yaml` kế thừa lịch200/warmup20/save10,
keep20. Tokenizer local mặc định là bản Qwen student đã stage; probe server
phải kiểm parity. Endpoint235B từ thông tin cũ cần probe lại, không coi là live
qualification. Recipe C còn trỏ eval-7 cũ và chưa có allocation mới; user mới
chọn B cho eval-6, không tự xếp C vào node này. Handoff hiện hành:
`tools/xtoken_all/HANDOFF-three-arms-r11.md`, đọc mục r15 trước lịch sử bên dưới.

`render.py` nhận `--gym-actor-python`, `--gym-venvs`, `--gym-source` trỏ các thành phần
đã provision. Gym import phải đúng source đã khai; config paths dùng đường dẫn
tuyệt đối. Adapter bật `skip_load_from_dotenv=true`: không cần copy/nạp
`nemo_gym_env.yaml`, không đưa secret của pipeline khác vào effective config.

r7 thay `worker_python` bằng `actor_python`, bỏ CLI `--gym-python` cũ. Actor
`nemo_rl.environments.nemo_gym.NemoGym` cần Torch/Transformers/Ray và Gym SDK;
đây là môi trường `PY_EXECUTABLES.NEMO_GYM` trong registry Super. Đường provision
gốc đặt venv ở `$NEMO_RL_VENV_DIR/nemo_rl.environments.nemo_gym.NemoGym/bin/python`.
Phải kiểm file/import trên node; không giả định nó tồn tại. Chỉ định interpreter
đã provision bằng CLI mới, không gọi nhánh tạo venv của helper gốc. Nếu deployment
dùng path khác, operator xác nhận đúng interpreter actor đã chạy; không tự fallback.

Ba service `vllm_model`, `simple_agent`, `general_qa` giữ venv riêng dưới
`--gym-venvs`. Proxy `vllm_model` không phải engine vLLM. Renderer/factory từ chối
actor Python trỏ vào service store. `gym_actor_runtime_env()` giữ cách merge
environment của `environments.utils.create_env`: kế thừa env, đặt VIRTUAL_ENV và
UV_PROJECT_ENVIRONMENT cho actor, ưu tiên source đã review trong PYTHONPATH;
không trộn site-packages. Cùng helper được dùng bởi preflight, Ray probe và runtime.
`super-runtime-lineage.json` chốt SHA của bốn file tham chiếu Super còn nguyên.

Interpreter giữ đường dẫn tuyệt đối theo CLI, không dereference symlink
`venv/bin/python` sang binary gốc. `.resolve()` tại vị trí này từng làm mất
venv/site-packages và gây lỗi thiếu Hydra trong bản G1 trước khi có snapshot Super;
r4 sửa lỗi cho cả code hiện tại. Không ghép nguyên site-packages của venv Gym
vào driver: log thực tế đã ghi nhận Transformers4.57 gặp huggingface-hub1.25.1
thay vì bản <1.0 yêu cầu. `runtime_probe.py` ghi executable/prefix/base_prefix.
Regression stdlib Linux ở `tools/xtoken_all/tests/check_venv_interpreter.py` chạy
Python qua symlink với pyvenv.cfg/marker tạm; không cài package.

Tên run tùy ý không xác định thuật toán: renderer này luôn xuất mode `on` và
backend `nemo_gym`, kể cả tên bắt đầu A-off. Đường off-policy thật dùng entrypoint
và config off-policy riêng, không gọi Gym để sinh response student.

Gym fork `8949f55` luôn validate endpoint judge lúc khởi động, kể cả khi
`should_use_judge=false`. Config G1 khai endpoint/model của teacher đã có cho
phép kiểm tra metadata; không gọi LLM judge để chấm đáp án. Nếu endpoint cần
auth mà validator Gym không hỗ trợ, dừng và báo đúng lỗi, không mở server mới.
Live Gym fork khác commit hoặc dirty thì cần so diff trước khi dùng kết luận này.

Nếu source Gym trên node khác contract, dùng source đã review trong packet ở
checkout mới `xtoken_all/ordures_ultra_v3_gym/` sau khi kiểm manifest, cùng các
sub-venv đã provision. PYTHONPATH đặt source này trước source Gym kế thừa, giữ các
runtime import path còn lại. Preflight kiểm import cả Gym parent và ba sub-venv.
Không yêu cầu đồng bộ toàn bộ 3rdparty hoặc tải lại ZIP lớn.

## Recovery r10: decode và checkpoint eval

Log A-on-11 mới cho thấy update66 hoàn tất rồi rollout67 dừng khi decode sampled
student IDs chứa U+FFFD, trước teacher scoring. Collator nay mask riêng mẫu không
thể giữ text cho cross-tokenizer alignment, ghi `XTOKEN_RESPONSE_MASK`, giữ nguyên
sampled IDs và train các mẫu hợp lệ còn lại. Same-tokenizer dùng IDs trực tiếp,
không decode rồi encode lại. Toàn batch bị mask vẫn dừng trước optimizer; không
update rỗng hoặc dùng lại rollout. Các structural error khác không bị nuốt.

Checkpoint safetensors Super có shard offsets và metadata ẩn `.hf_metadata`;
không chỉ đổi tên file để serve. `tools/xtoken_all/export_checkpoint.py` inspect
metadata/header rồi gọi hàm consolidate trong Automodel đã provision, tạo output
HF mới có config/tokenizer/index và kiểm load CPU theo `--validate-load`.
Hash trước/sau bảo vệ source; lỗi để FAILED. Native DCP converter cũ không phải
đường ghép sharded safetensors này. Handoff chi tiết và block E01/E02 nằm ở
`tools/xtoken_all/HANDOFF-recovery-r10.md`. Chưa có bằng chứng export/load checkpoint
thật trên node; file closure và fixture pass không phải numerical parity.

Renderer bắt buộc `--ray-address host:port`. Driver chỉ nối head đã chỉ định,
không fallback auto-start. Lease allocation phải trỏ đúng head đó. Các biến
`NEMO_GYM_CONFIG_DICT`/`NEMO_GYM_CONFIG_PATH` thừa hưởng bị từ chối vì SDK ưu tiên
chúng hơn initial config; operator dùng môi trường tiến trình sạch, không dump
hay copy giá trị của run khác. Config G1 cũ cần render lại; không dùng YAML/receipt cũ.

## Chuỗi helper cho agent vận hành

Tất cả helper nằm tại `tools/xtoken_all/`; dùng interpreter công ty đã có.
Không bootstrap dependency hay bật Docker local.

1. Đối chiếu active source với tarball, tạo source mới dưới `xtoken_all/ordures_ultra_v3`;
   giữ source gốc. Nếu thiếu bundled dependency, liên kết rõ thư mục 3rdparty
   đã có hoặc dùng PYTHONPATH đã provision, không tự clone/cài lại.
2. Chuyển DAPO bằng `convert_general_qa.py`, tạo smoke bằng `make_smoke.py`.
3. Render bằng `render.py`: tham số đầy đủ có trong `--help`; mọi path bắt buộc.
   `provenance.json` chốt hash code, data, projection và effective YAML.
4. Chạy `preflight.py --run-dir <run>` bằng `NRL_PYTHON`. Nó kiểm tra shard,
   tokenizer, projection qua loader thật của X-Token, budget từng prompt,
   source import/API FSDP/checkpoint của policy worker, API async của generation
   worker và config/venv Gym. Nó không khởi động server. Rerun vô hiệu receipt cũ
   trước khi import; lỗi giữa chừng không để lại trọn bộ gate thành công cũ.
   r6 kiểm package theo role: policy cần `nemo-automodel` và API FSDP/checkpoint;
   generation cần Torch/Transformers/Ray/vLLM, source đúng và API async. Không
   yêu cầu Automodel trong generation venv. Log G52/G53 cho thấy probe r5 đã
   dừng ở phép đọc metadata Automodel sau khi import generation worker thành công;
   đây là yêu cầu thừa của probe. Không cài/chia sẻ site-packages policy sang
   generation chỉ để vượt nó. Kiểm tra import/API chưa chứng minh refit/GPU pass.
   r7 chạy `--gym-only` bằng actor_python với cùng runtime env dùng khi tạo actor.
   Nó import package bình thường của actor `NemoGym`, `x_token.gym_compat`, SDK
   RunHelper/RolloutCollectionHelper và kiểm async-generator API. Ba sub-venv chạy
   `gym_service_probe.py`, import app/class thật của từng service; không import
   NeMo RL hay yêu cầu Torch trong proxy. Không chỉ nạp file helper bằng importlib.
   Receipt ghi executable/prefix/module của actor và ba app. Khi lỗi, subprocess
   hiện traceback; source-check chỉ ghi thành công sau khi toàn bộ import qua.
5. `probe_teacher.py --config <effective.yaml>`
   kiểm alias, ba ca tokenizer parity, một yêu cầu score nhỏ và cap top-20.
   r5 cho phép server báo đúng `0.31.0` dùng mặc định upstream `raw_logprobs`
   (cap mặc định 20), theo `teacher_probe_policy.yaml`. Không cần host lại chỉ vì
   lệnh serve thiếu hai flag. Đây là giả định được chấp nhận về custom deployment,
   không phải phép đo mode: receipt ghi `mode_basis=upstream_default_assumption`,
   `server_logprobs_mode_verified=false`, `mode_assumption_accepted=true`.
   Chỉ metadata hoặc giả định mặc định không đủ: scoring/parity thật vẫn bắt buộc.
   [Mặc định vLLM 0.31.0](https://docs.vllm.ai/en/v0.31.0/cli/serve/#--logprobs-mode).
   Nếu biết deployment override mode hoặc version khác, thêm tùy chọn
   `--server-mode-evidence <json>`. Evidence cần `teacher_url`, `served_model`,
   `logprobs_mode=raw_logprobs`, `verification_source` là
   `operator_checked_server_args` hoặc `server_reported_config`, `observed_at_utc`,
   `evidence_description` đã bỏ secret. Evidence trái mode bị từ chối, không
   fallback mặc định. Version lạ không tự pass; probability mass/echo không
   chứng minh mode. Operator có thể tắt default assumption trong policy YAML.
   Probe thất bại vô hiệu receipt thành công cũ. Launcher kiểm cùng policy hash.
6. Operator xác nhận node, bốn UUID và Ray ownership hiện tại. Sau đó dùng
   `check_allocation.py` với `--node-label`, bốn `--expected-gpu-uuids`,
   `--ray-address`, `--operator-owned-ray-verified`, `--output` mới.
   Helper không start/stop Ray. Nó đòi 1 Ray node/4 GPU và GPU mục tiêu không bận.
7. Export `XTOKEN_ALL_ROOT`, `XTOKEN_RUN_DIR`, `NRL_PYTHON`,
   `NEMO_RL_VENV_DIR`, `XTOKEN_ALLOCATION_RECEIPT`; chạy `launch.sh`.
   Trước launch chạy `ray_stream_probe.py --run-dir <run> --allocation-receipt <lease>`:
   chỉ tạo một actor CPU không model trên Ray đã xác nhận, kiểm async generator,
   ray.wait/get, thứ tự kết quả và dọn actor của probe. Nó không chạy task GPU.
   r7 đặt actor probe trong chính runtime env của NemoGym, import actor/SDK thật
   rồi mới kiểm stream. Receipt ghi actor_runtime và launcher đối chiếu cùng
   executable/prefix/source/API như preflight. Không khởi tạo model hay service.
   Launcher kiểm UUID/occupancy lại ngay trước chạy, allocation chưa quá 5 phút,
   các receipt preflight/teacher/Ray stream chưa quá `receipt_max_age_s`,
   hash source/data/projection/config; không `ray stop --force`.
8. `check_result.py --run-dir <run>` đòi exit driver và tee bằng 0, đúng hai
   update, loss/gradient hữu hạn và gradient khác 0, teacher scores đúng dạng,
   Gym rollout versions 0 rồi 1, response không rỗng, reward không vào loss.
   r8 còn kiểm file closure của checkpoint step1/step2 khi bật checkpointing.

## Kiểm chứng và giới hạn

Test nhẹ độc lập tensor: `tools/xtoken_all/tests/`. Với test archive-bound, đặt
`XTOKEN_REVIEWED_GYM_ZIP` trỏ ZIP Gym đi kèm packet. Chúng chạy các method biên
trích từ source thật của SDK với dependency double; không import toàn stack Gym.
Test CPU tensor/Ray mock: `tests/unit/algorithms/x_token/test_gym_rollout.py`.
Chạy thêm regression X-Token đang có và các test company processor/Gym/penalty
trong runtime công ty. Không gộp nhiều test suite resource server Gym có cùng
tên `test_app.py` vào một lệnh.

Test protocol/converter không chứng minh server HTTP, CUDA IPC, FSDP refit hay
GPU training. Chỉ sau khi vượt preflight và hai update trên node mới báo smoke
GPU pass; vẫn chưa có kết luận chất lượng, hội tụ, resume parity hoặc full train.
Same-tokenizer Qwen235B là nhánh so sánh sau; không đổi loss/backend giữa hai nhánh.
