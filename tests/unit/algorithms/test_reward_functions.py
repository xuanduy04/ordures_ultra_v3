

import pytest
import torch

from nemo_rl.algorithms.reward_functions import (
    RewardShapingConfig,
    _compute_n_gram_repetition_penalty,
    apply_reward_shaping,
)
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from tests.unit.algorithms.utils import create_mock_batch_with_responses


def test_reward_shaping_disabled():
    """Test that when reward shaping is disabled, rewards remain unchanged."""
    # Create batch with various response lengths
    batch = create_mock_batch_with_responses(
        num_samples=3, response_lengths=[10, 20, 30], initial_rewards=[1.0, 0.5, 0.8]
    )

    original_rewards = batch["total_reward"].clone()

    # Disabled reward shaping config
    config = RewardShapingConfig(
        enabled=False,
        overlong_buffer_length=5,
        overlong_buffer_penalty=0.1,
        max_response_length=25,
    )

    # Apply reward shaping
    result_batch = apply_reward_shaping(batch, config)

    # Rewards should remain unchanged
    assert torch.allclose(result_batch["total_reward"], original_rewards)
    assert result_batch is batch  # Should return the same batch object


def test_reward_shaping_no_penalties():
    """Test reward shaping when all responses are within acceptable length."""
    # Create batch where all responses are shorter than expected length
    batch = create_mock_batch_with_responses(
        num_samples=3,
        response_lengths=[10, 15, 18],  # All <= 20 (expected_response_length)
        initial_rewards=[1.0, 0.5, 0.8],
    )

    original_rewards = batch["total_reward"].clone()

    # Config: max_response_length=25, overlong_buffer_length=5 -> expected_response_length=20
    config = RewardShapingConfig(
        enabled=True,
        overlong_buffer_length=5,
        overlong_buffer_penalty=1.0,
        max_response_length=25,
    )

    # Apply reward shaping
    result_batch = apply_reward_shaping(batch, config)

    # Since no responses exceed expected length, rewards should remain unchanged
    assert torch.allclose(result_batch["total_reward"], original_rewards)


def test_reward_shaping_with_penalties():
    """Test reward shaping when responses exceed expected length and receive penalties."""
    # Create batch with responses of varying lengths
    batch = create_mock_batch_with_responses(
        num_samples=4,
        response_lengths=[10, 22, 25, 30],  # expected_response_length = 20
        initial_rewards=[1.0, 0.8, 0.6, 0.4],
    )

    # Config: max_response_length=25, overlong_buffer_length=5 -> expected_response_length=20
    config = RewardShapingConfig(
        enabled=True,
        overlong_buffer_length=5,
        overlong_buffer_penalty=0.5,
        max_response_length=25,
    )

    # Apply reward shaping
    result_batch = apply_reward_shaping(batch, config)

    # Calculate expected rewards manually
    # Response 0: length=10, exceed_length=10-20=-10 (no penalty, reward stays 1.0)
    # Response 1: length=22, exceed_length=22-20=2, penalty=min(-2/5*0.5, 0)=-0.2, reward=0.8-0.2=0.6
    # Response 2: length=25, exceed_length=25-20=5, penalty=min(-5/5*0.5, 0)=-0.5, reward=0.6-0.5=0.1
    # Response 3: length=30, exceed_length=30-20=10, penalty=min(-10/5*0.5, 0)=-1.0, reward=0.4-1.0=-0.6

    expected_rewards = torch.tensor([1.0, 0.6, 0.1, -0.6])
    assert torch.allclose(result_batch["total_reward"], expected_rewards, atol=1e-6)


def test_reward_shaping_missing_dapo_params_is_noop():
    """Test that missing DAPO config values result in a no-op (not an error)."""
    batch = create_mock_batch_with_responses(
        num_samples=1, response_lengths=[20], initial_rewards=[1.0]
    )
    original_rewards = batch["total_reward"].clone()

    # Test missing overlong_buffer_length
    config = RewardShapingConfig(
        enabled=True,
        overlong_buffer_length=None,
        overlong_buffer_penalty=0.1,
        max_response_length=25,
    )
    result_batch = apply_reward_shaping(batch, config)
    assert torch.allclose(result_batch["total_reward"], original_rewards)

    # Test missing overlong_buffer_penalty
    config["overlong_buffer_length"] = 5
    config["overlong_buffer_penalty"] = None
    result_batch = apply_reward_shaping(batch, config)
    assert torch.allclose(result_batch["total_reward"], original_rewards)

    # Test missing max_response_length
    config["overlong_buffer_penalty"] = 0.1
    config["max_response_length"] = None
    result_batch = apply_reward_shaping(batch, config)
    assert torch.allclose(result_batch["total_reward"], original_rewards)


def test_reward_shaping_missing_assistant_response():
    """Test that missing assistant response raises assertion error."""
    # Create a batch with only user messages (no assistant responses)
    message_logs = [
        [{"role": "user", "content": "Question", "token_ids": torch.tensor([1, 2, 3])}]
    ]

    batch = BatchedDataDict[DatumSpec](
        {
            "task_name": ["math"],
            "message_log": message_logs,
            "extra_env_info": [{}],
            "loss_multiplier": torch.ones(1),
            "total_reward": torch.tensor([1.0]),
        }
    )

    config = RewardShapingConfig(
        enabled=True,
        overlong_buffer_length=5,
        overlong_buffer_penalty=0.1,
        max_response_length=25,
    )

    with pytest.raises(
        AssertionError, match="Assistant response not found during reward shaping"
    ):
        apply_reward_shaping(batch, config)


def test_reward_shaping_mismatched_lengths():
    """Test that mismatched message_log and rewards lengths raise assertion error."""
    # Create batch with mismatched lengths
    batch = create_mock_batch_with_responses(
        num_samples=2, response_lengths=[10, 20], initial_rewards=[1.0, 0.5]
    )

    # Manually add an extra reward to create mismatch
    batch["total_reward"] = torch.tensor(
        [1.0, 0.5, 0.3]
    )  # 3 rewards but 2 message_logs

    config = RewardShapingConfig(
        enabled=True,
        overlong_buffer_length=5,
        overlong_buffer_penalty=0.1,
        max_response_length=25,
    )

    with pytest.raises(
        AssertionError,
        match="The number of messages in the batch must match the number of rewards",
    ):
        apply_reward_shaping(batch, config)


def test_stop_properly_penalty():
    """Test stop_properly_penalty_coef shifts rewards for truncated samples."""
    batch = create_mock_batch_with_responses(
        num_samples=4,
        response_lengths=[10, 20, 30, 40],
        initial_rewards=[1.0, 0.8, 0.6, 0.4],
    )
    batch["truncated"] = torch.tensor([False, True, False, True])

    config = RewardShapingConfig(enabled=True, stop_properly_penalty_coef=0.5)
    result_batch = apply_reward_shaping(batch, config)

    # Non-truncated unchanged, truncated penalized by subtracting coef
    expected_rewards = torch.tensor([1.0, 0.3, 0.6, -0.1])
    assert torch.allclose(result_batch["total_reward"], expected_rewards, atol=1e-6)


def test_stop_properly_penalty_boundary_coefs():
    """Test boundary values: coef=0 subtracts nothing, coef=1 subtracts 1.0."""
    # Test coef=0: truncated samples unchanged (subtract 0)
    batch = create_mock_batch_with_responses(
        num_samples=2, response_lengths=[10, 20], initial_rewards=[1.0, 0.5]
    )
    batch["truncated"] = torch.tensor([True, True])

    config = RewardShapingConfig(enabled=True, stop_properly_penalty_coef=0.0)
    result = apply_reward_shaping(batch, config)
    assert torch.allclose(result["total_reward"], torch.tensor([1.0, 0.5]), atol=1e-6)

    # Test coef=1: subtract 1.0 from truncated samples
    batch["total_reward"] = torch.tensor([1.0, 0.5])
    config["stop_properly_penalty_coef"] = 1.0
    result = apply_reward_shaping(batch, config)
    assert torch.allclose(result["total_reward"], torch.tensor([0.0, -0.5]), atol=1e-6)


def test_stop_properly_penalty_error_cases():
    """Test error handling for missing truncated field."""
    batch = create_mock_batch_with_responses(
        num_samples=2, response_lengths=[10, 20], initial_rewards=[1.0, 0.5]
    )

    # Missing truncated field
    config = RewardShapingConfig(enabled=True, stop_properly_penalty_coef=0.5)
    with pytest.raises(AssertionError, match="truncated field not found"):
        apply_reward_shaping(batch, config)


def test_n_gram_repetition_penalty_short_sequence():
    """N-gram penalty is zero when sequence is shorter than n."""
    token_ids = torch.tensor([1, 2, 3])
    penalty = _compute_n_gram_repetition_penalty(token_ids, n=5, threshold=5)
    assert penalty == 0.0


def test_n_gram_repetition_penalty_no_repeats():
    """N-gram penalty is zero when no n-grams are over-repeated."""
    token_ids = torch.arange(20, dtype=torch.long)  # all distinct tokens
    penalty = _compute_n_gram_repetition_penalty(token_ids, n=5, threshold=2)
    assert penalty == 0.0


def test_n_gram_repetition_penalty_all_same():
    """Penalty for a sequence of identical tokens."""
    token_ids = torch.ones(8, dtype=torch.long)
    penalty = _compute_n_gram_repetition_penalty(token_ids, n=5, threshold=2)
    # total_n_grams = 4, one unique over-repeated 5-gram, freq=4
    # fraction_over = 1/4 = 0.25, normalized_max = 4/(8/5) = 2.5
    # penalty = max(0.25, 2.5) = 2.5
    assert abs(penalty - 2.5) < 1e-6


def test_n_gram_repetition_penalty_high_threshold():
    """N-gram penalty is zero when threshold exceeds all frequencies."""
    token_ids = torch.tensor([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], dtype=torch.long)
    penalty = _compute_n_gram_repetition_penalty(token_ids, n=5, threshold=10)
    assert penalty == 0.0


def test_n_gram_penalty_integration():
    """Test N-gram penalty applied via apply_reward_shaping."""
    # Create a batch with a repetitive response
    batch = create_mock_batch_with_responses(
        num_samples=2,
        response_lengths=[10, 10],
        initial_rewards=[1.0, 0.5],
    )
    # Make the first response repetitive (all same tokens)
    batch["message_log"][0][1]["token_ids"] = torch.ones(10, dtype=torch.long)

    original_rewards = batch["total_reward"].clone()

    config = RewardShapingConfig(
        enabled=True,
        n_gram_size=5,
        n_gram_threshold=2,
        n_gram_repetition_weight=0.5,
    )
    result_batch = apply_reward_shaping(batch, config)

    # First sample: repetitive, should get penalty
    # Second sample: non-repetitive (arange), should get 0 penalty
    assert result_batch["total_reward"][0] < original_rewards[0]
    assert torch.allclose(result_batch["total_reward"][1], original_rewards[1])


def test_n_gram_and_dapo_combined():
    """Test N-gram and DAPO penalties applied together."""
    batch = create_mock_batch_with_responses(
        num_samples=1,
        response_lengths=[30],
        initial_rewards=[1.0],
    )
    # Make response repetitive
    batch["message_log"][0][1]["token_ids"] = torch.ones(30, dtype=torch.long)

    config = RewardShapingConfig(
        enabled=True,
        n_gram_size=5,
        n_gram_threshold=2,
        n_gram_repetition_weight=0.3,
        overlong_buffer_length=5,
        overlong_buffer_penalty=1.0,
        max_response_length=25,
    )
    result_batch = apply_reward_shaping(batch, config)

    # Both penalties should be applied; reward should decrease
    assert result_batch["total_reward"][0] < 1.0


def test_n_gram_disabled_by_default():
    """Test that N-gram penalty is skipped when not configured."""
    batch = create_mock_batch_with_responses(
        num_samples=1, response_lengths=[20], initial_rewards=[1.0]
    )
    original_rewards = batch["total_reward"].clone()

    config = RewardShapingConfig(enabled=True)
    result_batch = apply_reward_shaping(batch, config)

    assert torch.allclose(result_batch["total_reward"], original_rewards)
