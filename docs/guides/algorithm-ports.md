# Algorithm ports qua pipeline X-Token / Gym

Đây là các mode opt-in trên source r16. Các YAML pilot_200.yaml và
pilot_200_4b.yaml, checkpoint/run cũ, P-KL/Gold/H-KL, topk_tail_kl, local teacher
IPC và đường TP/CP legacy được giữ nguyên. Không đổi mode khi gặp capability
error. Mỗi thay đổi mode phải tạo run mới, không resume chéo thuật toán.

## Mode và score contract

| Mode | Teacher scores | Objective / giới hạn |
| --- | --- | --- |
| dpca | topk hoặc full, phải có actual prompt-token logprob | Semantic-prior chunk advantage và clipped policy objective; bắt buộc behavior logprob thật của token đã sample |
| ctopd_full | full | RKL điều kiện trên shared vocabulary, chỉ vị trí1:1; port từ Rethinking/CTOPD, alignment UTF-8 bytes |
| ctopd_topk | topk | Approximation riêng: chỉ shared candidates server đã trả tại vị trí1:1; không phải full-vocabulary CTOPD |
| simct_full | full | Virtual shared+span vocabulary, native logprob trước average (paper-score mode), GH mass guard |
| simct_topk | topk | Approximation riêng: shared candidates được trả + realized span; GH dùng mass của candidate space thu hẹp này |
| grass_chunk_fixed_sigma | topk hoặc full, actual prompt-token logprob | Fixed native alignment chunks, exact LM-head Gram, fixed sigma2 explicit; không thay cho running-noise/EMA implementation trong repo SimCT cũ |

Hậu tố _full nói về teacher-score contract, không chứng minh tái lập toàn paper.
Server top20 hiện tại không đủ scores cho ctopd_full/simct_full; phải chọn mode
_topk rõ ràng hoặc một endpoint đã có full scores. Không sửa/restart teacher.
Giới hạn max_score_elements/max_response_bytes của client vẫn có hiệu lực:
full scores ở vocabulary lớn có thể vượt giới hạn bộ nhớ/network. Không tự tăng
các giới hạn này hoặc suy full distribution từ top20.

Raw-logit virtual scores của author-code SimCT vẫn có kernel/reference test,
nhưng adapter remote chọn mean_logprob rõ ràng: prompt-logprob API không cho
phép khôi phục additive logit offsets ở mỗi autoregressive position. Không gọi
paper-score mode là bitwise reproduction của author raw-logit mode.

## Những phần tái dùng và thay đổi có chủ đích

Gym rollout, one-update version lease, teacher client, checkpoint manager,
allocation/Ray/source receipts, refit4GiB và dataset converter Super vẫn dùng
chung. Reward QA vẫn chỉ là metric, không thêm vào loss.

Variant aligner xây minimal synchronized chunks từ byte boundaries. Không xóa
whitespace, không decode/re-tokenize sampled student IDs. Alignment legacy vẫn
có trong TokenAligner; các native chunks của nó làm partition GRASS. Byte-codec
hỗ trợ ByteLevel và SentencePiece-compatible tokenizer; tokenizer khác báo lỗi.
Role EOS được ghép trong shared vocabulary, nhưng terminal/control tokens vẫn
không tham gia response loss theo collator cũ.

DPCA giữ generation_logprobs từ message Gym, cùng vị trí sampled token ID;
không dùng detach(current logprob) để giả lập behavior policy. Data exporter
giữ tensor này đến student loss. Thiếu actual teacher prompt-token score hoặc
behavior logprob phải báo lỗi. Padding không được đi qua exp/log-ratio.

GRASS bắt actual input của output head qua hook opt-in trong Automodel forward,
cộng bias coordinate nếu head có bias. Gram tính bằng inner products của hidden
và logit gradients, chỉ trong fixed native chunks. Không dùng identity/diagonal
Gram thay thế; không có DP/partition search. Statistics được detach, loss vẫn
backprop qua current token NLL. Fixed sigma2 là mode mới để tránh giả vờ đã port
running EMA cùng checkpoint state; muốn dùng EMA cần thêm mode/state contract
riêng. Native singleton chunks không có shrinkage; đây là hành vi dự kiến.

Gram này là gradient theo nhánh output head khi giữ hidden cố định, không phải
Gram gradient của toàn Transformer. Nếu Qwen buộc chung embedding/head weight,
nó không cộng đường gradient qua input embedding. Đây là mô hình hình học cục bộ
được chọn cho mode này; không gọi nó là full-model gradient Gram.

Các adapter mới ban đầu giới hạn một remote teacher, TP=CP=1, fixed microbatch,
không sequence packing. Các khả năng legacy rộng hơn vẫn đi qua loss cũ khi
không có loss_fn.variant. Không âm thầm chuyển local IPC teacher sang remote.

## 2, 3 và 4 GPU

Recipe tools/xtoken_all/pilot_algorithm.yaml giữ GBS12/MBS1 để chia hết cho cả
DP2/3/4, TP=CP=1. Đây là budget riêng của algorithm suite; không đổi GBS của
pilot1.7B/4B cũ. Giữ200 update,warmup20,save10,keep20 trong recipe.
So sánh với GBS4 cần baseline tương ứng;200 step GBS12 tiêu thụ nhiều mẫu hơn.
Loss mới chia theo tổng số student response tokens hợp lệ trên global batch,
kể cả khi CTOPD loại các span không phải1:1. Khác mẫu số kept-segments của một
số author implementations; cần khớp normalization/LR trước so sánh paper.

Trên source suite, đặt XTOKEN_PILOT_CONFIG tới pilot_algorithm.yaml rồi dùng
run_xtoken.sh cross hoặc same, thêm --gpu-indices 0 1 / 0 1 2 / 0 1 2 3.
Subset phải thuộc allocation UUID đã khai; không dò/chiếm GPU khác. Launcher
re-render cluster/worker counts và kiểm divisibility trước model loading.
Không dùng wrapper pilot_200.sh cũ để chạy variant recipe.

Recipe variant tắt hai optional DeepGEMM switches qua runtime.environment;
chỉ có hiệu lực cho run dùng recipe này. Nó giữ backend cũ của các recipe
legacy. Không rebuild venv, sửa glibc hoặc thay teacher. Generation runtime
probe nay phải import vllm._C thật để phát hiện editable source thiếu binary.

## Kiểm chứng và phạm vi bằng chứng

test_algorithm_variants.py kiểm objective/gradient, clipping, chunk accounting,
masking, SimCT virtual vocab/GH, GRASS head Gram và credit conservation. Fixtures
tham chiếu DPCA/SimCT giữ source SHA và MIT attribution trong tests/fixtures.

variant_ddp_probe.py chạy NCCL thật trên2/3/4 rank, so loss, gradient và một update
với reference global-batch12. Numeric cross-tokenizer fixtures bao gồm1:n/n:1,
padding và sample weights; đây không phải bằng chứng Gym/vLLM/FSDP2 end-to-end.
Kết quả remote và source snapshot của mỗi lần chạy phải được đọc cùng report.

Không gọi canary một vài update là hiệu quả khoa học hoặc replicate paper.
Server Gemma/Qwen công ty không truy cập được từ Modal; built /opt/nemo-rl18GiB
trên shared chưa được cung cấp qua HF. NGC image có digest + source overlay là
môi trường tương đương để kiểm logic, chưa phải bản sao bitwise của image công ty.
Không lấy CTOPD/DPCA runtime khác thay vào NeMo chỉ vì cùng B200/CUDA.

## Nguồn thuật toán

- DPCA: ivanniu/On-Policy-Distill, commit927a8264f2e303b7f82c2d331a58fd4240c8805a;
  scalar objective đối chiếu port _mp_opd_dpca.py trong source SimCT đã có.
- CTOPD/Rethinking: Cross-Tokenizer-OPD author import trong repo ctopd,
  SimpleCrossTokenizerKD và byte_align_ctkd. Port dùng byte-exact alignment,
  không khẳng định đồng nhất author whitespace-normalized alignment.
- SimCT: sunjie279/SimCT-, SpanCrossTokenizerKD; giữ phân biệt raw_logit và
  mean_logprob. Fixture provenance ghi hash source thực tế, không chỉ tên repo.
- GRASS-Chunk: source _mp_opd_grass_chunk.py/_mp_opd_grass_span.py trong SimCT;
  fixed chunks và local output-head risk model, không phải GRASS-DP/global optimum.
