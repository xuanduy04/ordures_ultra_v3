

import os
import subprocess
import tempfile
from pathlib import Path

import pytest


class TestGenerateFromHF:
    """
    Test text generation from HuggingFace models with different parallelism configurations.
    """

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,test_name",
        [
            (2, 1, "TP"),
            (1, 2, "PP"),
        ],
    )
    def test_generate_parallelism(self, tmp_path, tp, pp, test_name):
        """
        Test text generation with different parallelism configurations.

        Args:
            tmp_path: Pytest temporary path fixture
            tp: Tensor parallelism size
            pp: Pipeline parallelism size
            test_name: Name of the test for identification
        """
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp_file:
            output_file = tmp_file.name

        # Run hf_to_megatron_generate_text.py with specified parallelism configuration
        cmd = [
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--nnodes=1",
            "-m",
            "coverage",
            "run",
            "--data-file=/opt/Megatron-Bridge/.coverage",
            "--source=/opt/Megatron-Bridge/",
            "--parallel-mode",
            "examples/conversion/hf_to_megatron_generate_text.py",
            "--hf_model_path",
            "meta-llama/Llama-3.2-1B",
            "--prompt",
            "Hello, how are you?",
            "--max_new_tokens",
            "10",
            "--tp",
            str(tp),
            "--pp",
            str(pp),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent
            )

            # Write output to file for debugging
            with open(output_file, "w") as f:
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\nSTDERR:\n")
                f.write(result.stderr)

            # Check that the generation completed successfully
            assert "GENERATED TEXT OUTPUT" in result.stdout, (
                f"Generation output not found in {test_name} test log. Output: {result.stdout}"
            )

            # Check that the prompt appears in the output
            assert "Hello, how are you?" in result.stdout, (
                f"Original prompt not found in {test_name} test generation output. Output: {result.stdout}"
            )

            # Check that generated text is present (should contain more than just the prompt)
            assert "Generated:" in result.stdout, (
                f"Generated text section not found in {test_name} test output. Output: {result.stdout}"
            )

            print(f"SUCCESS: {test_name} test completed successfully")
            print(f"{test_name} generation output:")
            print(result.stdout)

        finally:
            # Clean up temporary file
            if os.path.exists(output_file):
                os.unlink(output_file)
