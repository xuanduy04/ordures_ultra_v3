

from tests.utils.test_utils import run_test_script

TEST_FOLDER = "hf_dcp"
DCP_FSDP2_CHECKPOINT_FILENAME = "L2_DCP_FSDP2_Checkpoint.sh"
DCP_VLM_FSDP2_CHECKPOINT_FILENAME = "L2_DCP_VLM_FSDP2_Checkpoint.sh"
HF_DCP_FSDP2_CHECKPOINT_FILENAME = "L2_HF_DCP_FSDP2_Checkpoint.sh"
HF_DCP_VLM_FSDP2_CHECKPOINT_FILENAME = "L2_HF_DCP_VLM_FSDP2_Checkpoint.sh"
HF_DCP_PP2_CHECKPOINT_FILENAME = "L2_DCP_PP2_Checkpoint.sh"
import shutil


class TestHFDCP:
    def test_dcp_fsdp2_checkpoint(self):
        try:
            run_test_script(TEST_FOLDER, DCP_FSDP2_CHECKPOINT_FILENAME)
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)

    def test_dcp_vlm_fsdp2_checkpoint(self):
        try:
            run_test_script(TEST_FOLDER, DCP_VLM_FSDP2_CHECKPOINT_FILENAME)
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)

    def test_hf_dcp_fsdp2_checkpoint(self):
        try:
            run_test_script(TEST_FOLDER, HF_DCP_FSDP2_CHECKPOINT_FILENAME)
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)

    def test_hf_dcp_vlm_fsdp2_checkpoint(self):
        try:
            run_test_script(TEST_FOLDER, HF_DCP_VLM_FSDP2_CHECKPOINT_FILENAME)
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)

    def test_hf_dcp_pp2_checkpoint(self):
        try:
            run_test_script(TEST_FOLDER, HF_DCP_PP2_CHECKPOINT_FILENAME)
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)
