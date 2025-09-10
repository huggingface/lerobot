#!/usr/bin/env python
# TODO(pepijn): delete this file
"""Quick test script to load and test only the PI0.5 model from HuggingFace hub."""

from test_pi0_hub import test_hub_loading


def main():
    """Test only the PI0.5 model."""
    print("\n")
    print("=" * 60)
    print("PI0.5 Model Quick Test")
    print("=" * 60)

    success = test_hub_loading(model_id="pepijn223/pi05_base_fp32", model_name="PI0.5")

    if success:
        print("\n✅ PI0.5 model loaded and tested successfully!")
    else:
        print("\n❌ PI0.5 test failed!")

    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
