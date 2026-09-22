"""
scripts/verify_environment.py
==============================
Phase 1 — Environment Verification Script

Checks that all required packages are installed and reports
version numbers, GPU availability, and CUDA details.

No model weights are loaded.
No dataset is read.
No training is performed.

Usage:
    python scripts/verify_environment.py

Expected output: all checks PASS before proceeding to Phase 2.
"""

import sys
import os

# Force UTF-8 output on Windows terminals that default to cp1252
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Terminal colour helpers (no external deps required — uses ANSI directly)
# ---------------------------------------------------------------------------
RESET  = "\033[0m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"


def _ok(msg: str) -> None:
    print(f"  {GREEN}[PASS]{RESET}  {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}[FAIL]{RESET}  {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}[WARN]{RESET}  {msg}")


def _section(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'-' * 55}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'-' * 55}{RESET}")


# ---------------------------------------------------------------------------
# 1. Python version
# ---------------------------------------------------------------------------
def check_python() -> bool:
    _section("Python")
    major, minor = sys.version_info[:2]
    version_str = f"{major}.{minor}.{sys.version_info[2]}"
    if major == 3 and minor >= 9:
        _ok(f"Python {version_str}")
        return True
    else:
        _fail(
            f"Python {version_str} detected — Python 3.9+ is recommended. "
            "Older versions may cause compatibility issues."
        )
        return False


# ---------------------------------------------------------------------------
# 2. PyTorch + CUDA
# ---------------------------------------------------------------------------
def check_torch() -> bool:
    _section("PyTorch / CUDA")
    all_ok = True

    try:
        import torch
        _ok(f"torch {torch.__version__}")

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                mem_total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                _ok(f"GPU {i}: {name}  ({mem_total:.1f} GB VRAM)")
            _ok(f"CUDA version (torch build): {torch.version.cuda}")
        else:
            _warn(
                "CUDA is NOT available — PyTorch will run on CPU. "
                "Training will be very slow without a GPU. "
                "If you have an NVIDIA GPU, reinstall PyTorch with the correct "
                "CUDA index URL (see requirements.txt)."
            )

        # Quick tensor op to verify torch is functional
        try:
            t = torch.zeros(2, 2)
            _ = t + 1
            _ok("Basic tensor operation successful.")
        except Exception as e:
            _fail(f"Basic tensor operation failed: {e}")
            all_ok = False

    except ImportError:
        _fail(
            "torch is NOT installed. "
            "Install it with: "
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        )
        all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# 3. torchvision
# ---------------------------------------------------------------------------
def check_torchvision() -> bool:
    _section("torchvision")
    try:
        import torchvision
        _ok(f"torchvision {torchvision.__version__}")
        return True
    except ImportError:
        _fail("torchvision is NOT installed (should be installed alongside torch).")
        return False


# ---------------------------------------------------------------------------
# 4. OpenCV
# ---------------------------------------------------------------------------
def check_opencv() -> bool:
    _section("OpenCV")
    try:
        import cv2
        _ok(f"cv2 (opencv) {cv2.__version__}")

        # Check that Farneback Optical Flow is available (it is in standard builds)
        has_farneback = hasattr(cv2, "calcOpticalFlowFarneback")
        if has_farneback:
            _ok("cv2.calcOpticalFlowFarneback is available (Farneback Optical Flow ready).")
        else:
            _fail(
                "cv2.calcOpticalFlowFarneback NOT found — "
                "try reinstalling opencv-python."
            )
            return False

        return True
    except ImportError:
        _fail("opencv-python is NOT installed.  pip install opencv-python")
        return False


# ---------------------------------------------------------------------------
# 5. NumPy
# ---------------------------------------------------------------------------
def check_numpy() -> bool:
    _section("NumPy")
    try:
        import numpy as np
        _ok(f"numpy {np.__version__}")
        arr = np.zeros((4, 4), dtype=np.float32)
        _ok(f"Basic array creation successful (shape={arr.shape}, dtype={arr.dtype}).")
        return True
    except ImportError:
        _fail("numpy is NOT installed.  pip install numpy")
        return False


# ---------------------------------------------------------------------------
# 6. Pillow
# ---------------------------------------------------------------------------
def check_pillow() -> bool:
    _section("Pillow (PIL)")
    try:
        from PIL import Image
        import PIL
        _ok(f"Pillow {PIL.__version__}")
        return True
    except ImportError:
        _fail("Pillow is NOT installed.  pip install Pillow")
        return False


# ---------------------------------------------------------------------------
# 7. Matplotlib
# ---------------------------------------------------------------------------
def check_matplotlib() -> bool:
    _section("Matplotlib")
    try:
        import matplotlib
        _ok(f"matplotlib {matplotlib.__version__}")
        return True
    except ImportError:
        _fail("matplotlib is NOT installed.  pip install matplotlib")
        return False


# ---------------------------------------------------------------------------
# 8. tqdm
# ---------------------------------------------------------------------------
def check_tqdm() -> bool:
    _section("tqdm")
    try:
        import tqdm
        _ok(f"tqdm {tqdm.__version__}")
        return True
    except ImportError:
        _fail("tqdm is NOT installed.  pip install tqdm")
        return False


# ---------------------------------------------------------------------------
# 9. scikit-learn
# ---------------------------------------------------------------------------
def check_sklearn() -> bool:
    _section("scikit-learn")
    try:
        import sklearn
        _ok(f"scikit-learn {sklearn.__version__}")
        return True
    except ImportError:
        _fail("scikit-learn is NOT installed.  pip install scikit-learn")
        return False


# ---------------------------------------------------------------------------
# 10. TensorBoard (optional)
# ---------------------------------------------------------------------------
def check_tensorboard() -> None:
    _section("TensorBoard (optional)")
    try:
        import tensorboard
        _ok(f"tensorboard {tensorboard.__version__}  [optional — available]")
    except ImportError:
        _warn("TensorBoard is NOT installed — training curve logging will be skipped. "
              "Install with: pip install tensorboard")


# ---------------------------------------------------------------------------
# 11. Project structure
# ---------------------------------------------------------------------------
def check_project_structure() -> bool:
    _section("Project Structure")
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    expected_dirs = [
        "modules",
        "models/unet",
        "utils",
        "scripts",
        "data/train",
        "data/val",
        "data/test",
        "outputs/masks",
        "outputs/optical_flow",
        "outputs/enhanced",
        "outputs/metrics",
        "checkpoints",
    ]
    expected_files = [
        "config.py",
        "requirements.txt",
        "README.md",
        ".gitignore",
    ]

    all_ok = True
    for d in expected_dirs:
        path = os.path.join(base, d)
        if os.path.isdir(path):
            _ok(f"Directory exists: {d}/")
        else:
            _fail(f"Directory MISSING: {d}/")
            all_ok = False

    for f in expected_files:
        path = os.path.join(base, f)
        if os.path.isfile(path):
            _ok(f"File exists: {f}")
        else:
            _fail(f"File MISSING: {f}")
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# 12. config.py import check
# ---------------------------------------------------------------------------
def check_config() -> bool:
    _section("config.py Import")
    try:
        # Add project root to path so config can be imported from scripts/
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        import config  # noqa: F401
        _ok("config.py imports successfully.")
        _ok(f"  BASE_DIR      = {config.BASE_DIR}")
        _ok(f"  INPUT_HEIGHT  = {config.INPUT_HEIGHT}")
        _ok(f"  INPUT_WIDTH   = {config.INPUT_WIDTH}")
        _ok(f"  LOSS_TYPE     = {config.LOSS_TYPE}")
        _ok(f"  SEED          = {config.SEED}")
        return True
    except Exception as e:
        _fail(f"config.py failed to import: {e}")
        return False


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"\n{BOLD}{'=' * 55}")
    print("  wildlife-COD -- Environment Verification")
    print(f"{'=' * 55}{RESET}")

    results = {
        "Python"           : check_python(),
        "PyTorch + CUDA"   : check_torch(),
        "torchvision"      : check_torchvision(),
        "OpenCV"           : check_opencv(),
        "NumPy"            : check_numpy(),
        "Pillow"           : check_pillow(),
        "Matplotlib"       : check_matplotlib(),
        "tqdm"             : check_tqdm(),
        "scikit-learn"     : check_sklearn(),
        "Project Structure": check_project_structure(),
        "config.py"        : check_config(),
    }
    check_tensorboard()  # optional — not counted in pass/fail

    _section("SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total  = len(results)

    for name, status in results.items():
        symbol = f"{GREEN}[OK]{RESET}" if status else f"{RED}[NO]{RESET}"
        print(f"  {symbol}  {name}")

    print()
    if passed == total:
        print(
            f"  {GREEN}{BOLD}ALL {total}/{total} CHECKS PASSED.{RESET}  "
            "Environment is ready -- proceed to Phase 2 (Dataset Inspection)."
        )
    else:
        failed = total - passed
        print(
            f"  {RED}{BOLD}{failed} CHECK(S) FAILED.{RESET}  "
            "Resolve the issues above before proceeding."
        )

    print()


if __name__ == "__main__":
    main()