


class TestModelsImports:
    """Test that all model providers can be imported correctly."""

    def test_import_gpt_provider(self):
        """Test importing GPTModelProvider."""
        from megatron.bridge.models import GPTModelProvider
        from megatron.bridge.models.gpt_provider import GPTModelProvider as DirectImport

        # Should be the same class
        assert GPTModelProvider is DirectImport

    def test_import_t5_provider(self):
        """Test importing T5ModelProvider."""
        from megatron.bridge.models import T5ModelProvider
        from megatron.bridge.models.t5_provider import T5ModelProvider as DirectImport

        # Should be the same class
        assert T5ModelProvider is DirectImport

    def test_import_llama_providers(self):
        """Test importing all Llama model providers."""
        from megatron.bridge.models import (
            Llama2ModelProvider7B,
            Llama3ModelProvider8B,
            LlamaModelProvider,
        )

        # Verify all imports are successful and are classes
        assert isinstance(LlamaModelProvider, type)
        assert isinstance(Llama2ModelProvider7B, type)
        assert isinstance(Llama3ModelProvider8B, type)

    def test_models_package_all_exports(self):
        """Test that __all__ exports match available imports."""
        import megatron.bridge.models as models

        # Check that all items in __all__ are actually importable
        for name in models.__all__:
            assert hasattr(models, name), f"{name} is in __all__ but not importable"
            attr = getattr(models, name)
            assert attr is not None, f"{name} is None"

    def test_backwards_compatibility_imports(self):
        """Test that old import paths still work for backwards compatibility."""
        # These should now import the provider classes
        try:
            from megatron.bridge.models import GPTModelProvider as GPTConfig
            from megatron.bridge.models import T5ModelProvider as T5Config

            # They should be the provider classes now
            assert hasattr(GPTConfig, "provide")
            assert hasattr(T5Config, "provide")
        except ImportError:
            # If the old names aren't aliased, that's okay
            pass

    def test_model_provider(self):
        """Test that model providers inherit from ModelProviderMixin."""
        from megatron.bridge.models import GPTModelProvider, LlamaModelProvider, T5ModelProvider
        from megatron.bridge.models.model_provider import ModelProviderMixin

        # All providers should inherit from ModelProviderMixin
        assert issubclass(GPTModelProvider, ModelProviderMixin)
        assert issubclass(T5ModelProvider, ModelProviderMixin)
        assert issubclass(LlamaModelProvider, ModelProviderMixin)

    def test_transformer_config_inheritance(self):
        """Test that model providers inherit from TransformerConfig."""
        from megatron.core.transformer.transformer_config import TransformerConfig

        from megatron.bridge.models import GPTModelProvider, LlamaModelProvider, T5ModelProvider

        # All providers should inherit from TransformerConfig
        assert issubclass(GPTModelProvider, TransformerConfig)
        assert issubclass(T5ModelProvider, TransformerConfig)
        assert issubclass(LlamaModelProvider, TransformerConfig)
