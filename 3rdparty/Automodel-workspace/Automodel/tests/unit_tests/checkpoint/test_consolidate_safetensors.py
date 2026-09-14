

import os

import pytest

from nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors import (
    _write_sub_tensor_to_file_optimized,
)


@pytest.mark.run_only_on("CPU")
def test_write_scalar_tensor(tmp_path):
    """Ensure that a 0-dim (scalar) tensor shard is written to the output file.

    Regression test for a bug where `_write_sub_tensor_to_file_optimized` used to
    early-return on ``tensor_shape == []`` and therefore omitted scalar payloads,
    which produced corrupt `.safetensors` files (incomplete metadata).
    """

    # Prepare an empty temporary output file
    output_file = tmp_path / "scalar_tensor.bin"
    output_file.write_bytes(b"")  # create the file

    # Fake scalar tensor payload (2-byte BF16 value)
    sub_tensor_bytes = b"\x34\x12"
    element_size = len(sub_tensor_bytes)  # 2 bytes for BF16

    # Prepare destination buffer for a scalar tensor (element_size bytes)
    full_tensor_mv = memoryview(bytearray(element_size))

    # Call the routine under test: scalar has empty shapes and offsets
    _write_sub_tensor_to_file_optimized(
        full_tensor_mv,
        sub_tensor_bytes,
        element_size,
        tensor_shape=[],  # scalar
        sub_tensor_offsets=[],
        sub_tensor_shape=[],
    )

    # Emulate file write as done by the caller in production code
    output_file.write_bytes(full_tensor_mv.tobytes())

    # The file must now contain exactly the scalar payload
    written = output_file.read_bytes()
    assert written == sub_tensor_bytes
    assert os.path.getsize(output_file) == element_size
