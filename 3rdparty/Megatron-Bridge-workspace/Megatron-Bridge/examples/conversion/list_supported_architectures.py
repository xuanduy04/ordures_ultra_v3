

from megatron.bridge import AutoBridge


def main() -> None:
    """List all HuggingFace model architectures supported by the AutoBridge."""
    supported_models = AutoBridge.list_supported_models()

    print("🚀 Megatron-Bridge AutoBridge - Supported Models")
    print("=" * 50)
    print()

    if not supported_models:
        print("❌ No supported models found.")
        print("   This might indicate that no bridge implementations are registered.")
        return

    print(f"✅ Found {len(supported_models)} supported model architecture(s):")
    print()

    for i, model in enumerate(supported_models, 1):
        print(f"  {i:2d}. {model}")

    print()
    print("💡 Usage:")
    print("   To use any of these models, you can load them with:")
    print("   >>> bridge = AutoBridge.from_hf_pretrained('model_name')")
    print("   >>> model = bridge.to_megatron_model()")
    print()
    print("🔍 Model Bridge Details:")
    print("   Each model has specific implementation details and configurations.")
    print("   Check the src/megatron/bridge/models/ directory for:")
    print("   • Model-specific bridge implementations")
    print("   • Configuration examples and README files")
    print("   • Weight mapping details")
    print("   • Architecture-specific optimizations")
    print()
    print("📚 For more examples, see the examples/bridge/ directory.")


if __name__ == "__main__":
    main()
