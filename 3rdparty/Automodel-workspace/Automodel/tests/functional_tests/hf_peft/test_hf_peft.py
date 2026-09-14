

from tests.utils.test_utils import run_test_script
import shutil

TEST_FOLDER = "hf_peft"
HF_PEFT_FSDP2_CHECKPOINT_FILENAME = "L2_HF_PEFT_FSDP2_Checkpoint.sh"
HF_PEFT_FSDP2_CHECKPOINT_QAT_FILENAME = "L2_HF_PEFT_FSDP2_Checkpoint_qat.sh"
HF_PEFT_Triton_FSDP2_CHECKPOINT_FILENAME = "L2_HF_PEFT_Triton_FSDP2_Checkpoint.sh"
HF_PEFT_VLM_FSDP2_CHECKPOINT_FILENAME = "L2_HF_PEFT_VLM_FSDP2_Checkpoint.sh"

class TestHFPEFT:
    def test_hf_peft_fsdp2_checkpoint(self):
        try:
            run_test_script(TEST_FOLDER, HF_PEFT_FSDP2_CHECKPOINT_FILENAME)
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)

    def test_hf_peft_fsdp2_checkpoint_qat(self):
        try:
            run_test_script(TEST_FOLDER, HF_PEFT_FSDP2_CHECKPOINT_QAT_FILENAME)
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)

    def test_hf_peft_triton_fsdp2_checkpoint(self):
        try:
            run_test_script(TEST_FOLDER, HF_PEFT_Triton_FSDP2_CHECKPOINT_FILENAME)
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)
    def test_hf_peft_vlm_fsdp2_checkpoint(self):
        try:
            run_test_script(TEST_FOLDER, HF_PEFT_VLM_FSDP2_CHECKPOINT_FILENAME)
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)