

from tests.utils.test_utils import run_test_script
import shutil

TEST_FOLDER = "hf_transformer_finetune"
HF_TRANSFORMER_SFT_FILENAME = "L2_HF_Transformer_SFT.sh"
HF_TRANSFORMER_SFT_MegatronFSDP_FILENAME = "L2_HF_Transformer_SFT_MegatronFSDP.sh"
HF_TRANSFORMER_PEFT_FILENAME = "L2_HF_Transformer_PEFT.sh"
HF_TRANSFORMER_PEFT_MegatronFSDP_FILENAME = "L2_HF_Transformer_PEFT_MegatronFSDP.sh"
HF_TRANSFORMER_PEFT_NO_TOKENIZER_FILENAME = "L2_HF_Transformer_PEFT_no_tokenizer.sh"
HF_TRANSFORMER_SFT_NO_LOGITS_FILENAME = "L2_HF_Transformer_SFT_no_logits.sh"
HF_TRANSFORMER_QWEN3_MOE_CUSTOM_FILENAME = "L2_HF_Transformer_Qwen3_MoE_custom.sh"
HF_TRANSFORMER_LLAMA3_CUSTOM_FILENAME = "L2_HF_Transformer_PEFT_Benchmark_Llama_custom.sh"
HF_TRANSFORMER_QWEN2_CUSTOM_FILENAME = "L2_HF_Transformer_PEFT_Benchmark_qwen2_custom.sh"

class TestHFTransformerFinetune:
    def test_hf_transformer_sft(self):
        try:
            run_test_script(TEST_FOLDER, HF_TRANSFORMER_SFT_FILENAME)
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)

    def test_hf_transformer_sft_megatron_fsdp(self):
        try:
            run_test_script(TEST_FOLDER, HF_TRANSFORMER_SFT_MegatronFSDP_FILENAME)
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)

    def test_hf_transformer_peft(self):
        try:
            run_test_script(TEST_FOLDER, HF_TRANSFORMER_PEFT_FILENAME)
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)

    def test_hf_transformer_peft_megatron_fsdp(self):
        try:
            run_test_script(TEST_FOLDER, HF_TRANSFORMER_PEFT_MegatronFSDP_FILENAME)
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)

    def test_hf_transformer_peft_no_tokenizer(self):
        try:
            run_test_script(TEST_FOLDER, HF_TRANSFORMER_PEFT_NO_TOKENIZER_FILENAME)
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)

    def test_hf_transformer_sft_no_logits(self):
        try:
            run_test_script(TEST_FOLDER, HF_TRANSFORMER_SFT_NO_LOGITS_FILENAME)
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)

    def test_hf_transformer_qwen3_moe_sdpa(self):
        try:
            run_test_script(TEST_FOLDER, HF_TRANSFORMER_QWEN3_MOE_CUSTOM_FILENAME)
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)

    def test_hf_transformer_llama3_custom(self):
        try:
            run_test_script(TEST_FOLDER, HF_TRANSFORMER_LLAMA3_CUSTOM_FILENAME)
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)

    def test_hf_transformer_qwen2_custom(self):
        try:
            run_test_script(TEST_FOLDER, HF_TRANSFORMER_QWEN2_CUSTOM_FILENAME)
        finally:
            # remove the checkpoint directory
            shutil.rmtree("checkpoints/", ignore_errors=True)