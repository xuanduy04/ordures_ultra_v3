# DSH — r11: giữ đủ20 checkpoint của pilot

User đã yêu cầu giữ nhiều checkpoint để eval các mốc. Retention3 trước đó là
lựa chọn của agent, không phải yêu cầu của user. Đã đổi cả train YAML cross
(same kế thừa) và recipe `pilot_200.yaml`:

- max_num_steps200, warmup20, save_period10 không đổi.
- keep_top_k20: giữ đủ step10,20,...,200, gồm optimizer theo recipe đã chốt.
- Không tự giảm retention hoặc xóa checkpoint để tiết kiệm dung lượng.
- Không đổi dataset, teacher, sampling, context hoặc LR vì cập nhật retention.

## Áp dụng

Nếu run chưa render: dùng source mới và helper như bàn giao pilot200; xác nhận
effective.yaml có checkpointing.keep_top_k=20 trước launch.

Nếu run đã render nhưng chưa launch: đổi recipe gốc rồi render run ID mới,
chạy lại preflight/teacher/allocation đúng source/config. Không sửa config/hash
để giả vờ các receipt cũ còn hợp lệ. Wrapper pilot_200.sh dùng run ID trong
pilot_200.yaml; khi dùng --run-id khác, dùng direct helpers với XTOKEN_RUN_DIR
đúng run mới thay vì wrapper trỏ nhầm run mặc định.

Nếu run đã chạy: CheckpointManager đã nạp retention vào RAM. Sửa YAML trên đĩa
không làm manager đổi3→20; không tuyên bố run hiện tại đã áp dụng. Không tự
stop/restart hoặc sửa source đang được job import. DSH cần giữ bản sao các
checkpoint còn lại trong thư mục archive ngoài checkpoint_dir trước các lần
prune tiếp; xác nhận source checkpoint đã finalize. Tái dùng lệnh snapshot đã
được operator xác minh, không di chuyển/xóa checkpoint nguồn. Snapshot có thể
dùng hardlink khi cùng filesystem và checkpoint đã bất biến; nếu không, dùng
copy vào persistent storage và kiểm đủ file, không link tới path sẽ bị xóa.

Các checkpoint đã prune không thể phục hồi chỉ bằng đổi YAML. Không suy ra
file đã mất nếu chưa list path/backup thật.

## Bằng chứng

Test offline thực thi chính method remove_old_checkpoints của Super với20 mốc
step10..200 và keep_top_k20, xác nhận không gọi rmtree checkpoint nào.
Source sửa nhỏ; không chạy GPU, không deploy hoặc sửa teacher. Export step20
A-on-11 đã pass theo log user trước đó, không cần convert lại vì retention.

Gói r11 có patch từ r10 cùng full source. Giữ các gói r10 và receipt cũ để
đối chiếu; không ghi đè artifact đã publish. Handoff training/recovery cũ vẫn
được kèm để tra đúng runtime, node và các gate.
