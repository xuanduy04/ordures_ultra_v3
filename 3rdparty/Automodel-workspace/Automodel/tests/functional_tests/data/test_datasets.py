

from tests.utils.test_utils import run_test_script

TEST_FOLDER = "data"
MEGATRON_BLENDED_DATASET_CHECKPOINT_TEST_FILENAME = "L2_Megatron_BlendedDataset_Checkpoint.sh" 
MEGATRON_SINGLE_DATASET_CHECKPOINT_TEST_FILENAME = "L2_Megatron_SingleDataset_Checkpoint.sh" 
MEGATRON_DP_SHARDING_TEST_FILENAME = "L2_Megatron_DP_Sharding_Test.sh"
MEGATRON_TP_SHARDING_TEST_FILENAME = "L2_Megatron_TP_Sharding_Test.sh"
MEGATRON_PREPROCESS_DATA_TEST_FILENAME = "L2_Megatron_Preprocess_Data.sh"

class TestDatasets:
    def test_megatron_blended_dataset_checkpoint(self):
        run_test_script(TEST_FOLDER, MEGATRON_BLENDED_DATASET_CHECKPOINT_TEST_FILENAME)

    def test_megatron_single_dataset_checkpoint(self):
        run_test_script(TEST_FOLDER, MEGATRON_SINGLE_DATASET_CHECKPOINT_TEST_FILENAME)

    def test_megatron_dp_sharding(self):
        run_test_script(TEST_FOLDER, MEGATRON_DP_SHARDING_TEST_FILENAME)

    def test_megatron_tp_sharding(self):
        run_test_script(TEST_FOLDER, MEGATRON_TP_SHARDING_TEST_FILENAME)

    def test_megatron_preprocess_data(self):
        run_test_script(TEST_FOLDER, MEGATRON_PREPROCESS_DATA_TEST_FILENAME)