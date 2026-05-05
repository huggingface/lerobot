#!/usr/bin/env python3
import sys
import os
import importlib
import subprocess

REQUIRED_PACKAGES = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("safetensors", "safetensors"),
    ("transformers", "transformers"),
    ("PIL", "Pillow"),
    ("matplotlib", "matplotlib"),
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
    ("einops", "einops"),
]

OPTIONAL_PACKAGES = [
    ("scipy", "scipy"),
    ("tqdm", "tqdm"),
]

MIN_VRAM_GB = 8.0
MODEL_SIZE_GB = 1.76
GRADIENT_OVERHEAD_FACTOR = 3.5


def check_python():
    v = sys.version_info
    print(f"Python: {v.major}.{v.minor}.{v.micro}")
    if v.major < 3 or (v.major == 3 and v.minor < 9):
        print("  [WARN] Python >= 3.9 recommended")
        return False
    print("  [OK]")
    return True


def check_packages(packages, required=True):
    missing = []
    for import_name, pip_name in packages:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  {import_name}: {version} [OK]")
        except ImportError:
            tag = "MISSING" if required else "OPTIONAL"
            print(f"  {import_name}: [{tag}]")
            missing.append(pip_name)
    return missing


def check_cuda():
    try:
        import torch
    except ImportError:
        print("  torch not installed, skipping CUDA check")
        return False

    print(f"  torch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("  [FAIL] No CUDA device found")
        return False

    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  cuDNN version: {torch.backends.cudnn.version()}")
    print(f"  GPU count: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / (1024 ** 3)
        free_mem, total_mem = torch.cuda.mem_get_info(i)
        free_gb = free_mem / (1024 ** 3)

        print(f"\n  GPU {i}: {props.name}")
        print(f"    Compute capability: {props.major}.{props.minor}")
        print(f"    Total VRAM: {total_gb:.1f} GB")
        print(f"    Free VRAM: {free_gb:.1f} GB")
        print(f"    SM count: {props.multi_processor_count}")

        estimated_need = MODEL_SIZE_GB * GRADIENT_OVERHEAD_FACTOR
        if free_gb < MIN_VRAM_GB:
            print(f"    [WARN] Free VRAM ({free_gb:.1f}GB) < {MIN_VRAM_GB}GB minimum")
        elif free_gb < estimated_need:
            print(f"    [WARN] Free VRAM ({free_gb:.1f}GB) may be tight for Grad-CAM (est. {estimated_need:.1f}GB)")
        else:
            print(f"    [OK] Sufficient VRAM for XAI ({estimated_need:.1f}GB estimated)")

    print(f"\n  bfloat16 support: {torch.cuda.is_bf16_supported()}")
    return True


def check_model_files(model_dir):
    print(f"\n  Model directory: {model_dir}")
    if not os.path.isdir(model_dir):
        print("  [FAIL] Directory not found")
        return False

    required_files = ["config.json", "model.safetensors"]
    all_found = True
    for f in required_files:
        path = os.path.join(model_dir, f)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 ** 2)
            print(f"  {f}: {size_mb:.1f} MB [OK]")
        else:
            print(f"  {f}: [MISSING]")
            all_found = False

    optional_files = [
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "train_config.json",
    ]
    for f in optional_files:
        path = os.path.join(model_dir, f)
        if os.path.exists(path):
            print(f"  {f}: [OK]")
        else:
            print(f"  {f}: [not found, optional]")

    return all_found


def check_test_image(image_path):
    print(f"\n  Test image: {image_path}")
    if not os.path.exists(image_path):
        print("  [WARN] Not found — you'll need a test image for XAI scripts")
        return False

    try:
        from PIL import Image
        img = Image.open(image_path)
        print(f"  Size: {img.size}")
        print(f"  Mode: {img.mode}")
        print("  [OK]")
        return True
    except Exception as e:
        print(f"  [FAIL] Cannot open: {e}")
        return False


def check_source_files(source_dir):
    print(f"\n  Source directory: {source_dir}")
    if not os.path.isdir(source_dir):
        print("  [FAIL] Directory not found")
        return False

    key_files = [
        "modeling_xvla.py",
        "modeling_florence2.py",
        "soft_transformer.py",
        "configuration_xvla.py",
        "configuration_florence2.py",
        "action_hub.py",
        "processor_xvla.py",
    ]
    for f in key_files:
        path = os.path.join(source_dir, f)
        status = "[OK]" if os.path.exists(path) else "[MISSING]"
        print(f"  {f}: {status}")

    return True


def print_install_commands(missing):
    if not missing:
        return
    pkgs = " ".join(missing)
    print(f"\n  pip install {pkgs}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    model_dir = os.path.join(project_dir, "xvla-pouring-0.1")
    source_dir = os.path.join(project_dir, "XVLA original source")
    test_image = os.path.join(project_dir, "test_image.jpg")

    print("=" * 60)
    print("XVLA XAI — Environment Check")
    print("=" * 60)

    print(f"\nOS: {sys.platform}")
    print(f"Project dir: {project_dir}")

    print("\n--- Python ---")
    check_python()

    print("\n--- Required Packages ---")
    missing_req = check_packages(REQUIRED_PACKAGES, required=True)

    print("\n--- Optional Packages ---")
    missing_opt = check_packages(OPTIONAL_PACKAGES, required=False)

    print("\n--- CUDA / GPU ---")
    cuda_ok = check_cuda()

    print("\n--- Model Files ---")
    model_ok = check_model_files(model_dir)

    print("\n--- Source Files ---")
    check_source_files(source_dir)

    print("\n--- Test Image ---")
    check_test_image(test_image)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if missing_req:
        print(f"\n[ACTION REQUIRED] Install missing packages:")
        print_install_commands(missing_req)
    else:
        print("\n[OK] All required packages installed")

    if missing_opt:
        print(f"\n[OPTIONAL] Install for enhanced features:")
        print_install_commands(missing_opt)

    if not cuda_ok:
        print("\n[FAIL] CUDA not available — XAI scripts require GPU")
    else:
        print("\n[OK] CUDA ready")

    if not model_ok:
        print("\n[FAIL] Model files incomplete")
    else:
        print("\n[OK] Model files ready")

    all_ok = len(missing_req) == 0 and cuda_ok and model_ok
    print(f"\n{'=' * 60}")
    if all_ok:
        print("READY — Proceed to load_model_test.py")
    else:
        print("NOT READY — Fix issues above before proceeding")
    print(f"{'=' * 60}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
