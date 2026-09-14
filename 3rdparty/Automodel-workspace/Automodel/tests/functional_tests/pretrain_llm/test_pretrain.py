

from tests.utils.test_utils import run_test_script
import shutil

TEST_FOLDER = "pretrain_llm"

class TestPretrainLLM:
    def test_pretrain(self):
        try:
            run_test_script(TEST_FOLDER, "L2_Pretrain.sh")
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)

    def test_pretrain_hf(self):
        try:
            run_test_script(TEST_FOLDER, "L2_Pretrain_HF.sh")
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)

    def test_pretrain_moonlight_16b_te_2l(self):
        try:
            run_test_script(TEST_FOLDER, "L2_Pretrain_Moonlight_16B_TE_2L.sh")
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)
