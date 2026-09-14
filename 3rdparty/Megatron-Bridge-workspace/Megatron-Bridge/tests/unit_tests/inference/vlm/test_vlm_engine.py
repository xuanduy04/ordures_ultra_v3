

from unittest.mock import MagicMock

from megatron.bridge.inference.vlm.vlm_engine import VLMEngine


class TestVLMEngine:
    def test_generate(self):
        mock_controller = MagicMock()
        mock_controller.tokenize_prompt.return_value = ([1, 2, 3], "image_dict")
        # Fix for TypeError: '>' not supported between instances of 'int' and 'MagicMock'
        mock_controller.inference_wrapped_model.context.max_batch_size = 128

        engine = VLMEngine(mock_controller, max_batch_size=4)
        engine.scheduler = MagicMock()
        engine.scheduler.add_request.return_value = "req_id"
        engine.scheduler.completed_request_pool = {"req_id": "result"}
        engine.run_engine = MagicMock()

        results = engine.generate(["prompt"], ["image"])

        assert results == ["result"]
        engine.scheduler.add_request.assert_called()
        engine.run_engine.assert_called()
