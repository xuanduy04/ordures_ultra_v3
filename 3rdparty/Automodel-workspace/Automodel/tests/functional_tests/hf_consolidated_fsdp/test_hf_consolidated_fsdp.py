

from tests.utils.test_utils import run_test_script

TEST_FOLDER = "hf_consolidated_fsdp"
HF_CONSOLIDATED_FSDP2_LLM_FILENAME = "L2_HF_Consolidated_FSDP2_LLM_Checkpoint.sh"
HF_CONSOLIDATED_FSDP2_VLM_FILENAME = "L2_HF_Consolidated_FSDP2_VLM_Checkpoint.sh"
HF_CONSOLIDATED_FSDP2_LLM_SCALAR_WEIGHT_FILENAME = "L2_HF_Consolidated_FSDP2_LLM_Checkpoint_Scalar_Param.sh"
HF_CONSOLIDATED_PP2_LLM_FILENAME = "L2_HF_Consolidated_PP2_LLM_Checkpoint.sh"

class TestHFConsolidatedFSDP:
    def test_hf_consolidated_fsdp2_llm_checkpoint(self):
        run_test_script(TEST_FOLDER, HF_CONSOLIDATED_FSDP2_LLM_FILENAME)

    def test_hf_consolidated_fsdp2_vlm_checkpoint(self):
        run_test_script(TEST_FOLDER, HF_CONSOLIDATED_FSDP2_VLM_FILENAME)
    
    def test_hf_consolidated_fsdp2_llm_checkpoint_scalar_weight(self):
        run_test_script(TEST_FOLDER, HF_CONSOLIDATED_FSDP2_LLM_SCALAR_WEIGHT_FILENAME)

    def test_hf_consolidated_pp2_llm_checkpoint(self):
        run_test_script(TEST_FOLDER, HF_CONSOLIDATED_PP2_LLM_FILENAME)