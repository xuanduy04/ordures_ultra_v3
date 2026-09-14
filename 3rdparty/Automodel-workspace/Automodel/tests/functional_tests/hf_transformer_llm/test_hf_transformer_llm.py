

import pytest

from tests.utils.test_utils import run_test_script

TEST_FOLDER = "hf_transformer_llm"
HF_TRANSFORMER_LLM_FSDP2_TP2_FILENAME = "L2_HF_Transformer_LLM_FSDP2_TP2.sh"
HF_TRANSFORMER_LLM_FSDP2_TP2_HF_TPPLAN_FILENAME = "L2_HF_Transformer_LLM_FSDP2_TP2_HF_TPPLAN.sh"
HF_TRANSFORMER_LLM_MegatronFSDP_TP2_FILENAME = "L2_HF_Transformer_LLM_MegatronFSDP_TP2.sh"
HF_TRANSFORMER_LLM_MegatronFSDP_TP2_HF_TPPLAN_FILENAME = "L2_HF_Transformer_LLM_MegatronFSDP_TP2_HF_TPPLAN.sh"
HF_TRANSFORMER_LLM_DDP_FILENAME = "L2_HF_Transformer_LLM_DDP.sh"


class TestHFTransformerLLM:
    def test_hf_transformer_llm_ddp(self):
        run_test_script(TEST_FOLDER, HF_TRANSFORMER_LLM_DDP_FILENAME)

    @pytest.mark.pleasefixme
    def test_hf_transformer_llm_fsdp2_tp2(self):
        run_test_script(TEST_FOLDER, HF_TRANSFORMER_LLM_FSDP2_TP2_FILENAME)
    
    @pytest.mark.pleasefixme
    def test_hf_transformer_llm_fsdp2_tp2_hf_tpplan(self):
        run_test_script(TEST_FOLDER, HF_TRANSFORMER_LLM_FSDP2_TP2_HF_TPPLAN_FILENAME)

    # @pytest.mark.pleasefixme
    # def test_hf_transformer_llm_megatron_fsdp_tp2(self):
    #     run_test_script(TEST_FOLDER, HF_TRANSFORMER_LLM_MegatronFSDP_TP2_FILENAME)
    
    # @pytest.mark.pleasefixme
    # def test_hf_transformer_llm_megatron_fsdp_tp2_hf_tpplan(self):
    #     run_test_script(TEST_FOLDER, HF_TRANSFORMER_LLM_MegatronFSDP_TP2_HF_TPPLAN_FILENAME)
