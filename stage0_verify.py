"""
Stage 0 verification script.
Run as: python stage0_verify.py
"""
import sys
import os

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding="utf-8")

FAILURES = []

def check(label, fn):
    try:
        fn()
        print(f"  [OK]  {label}")
    except Exception as e:
        print(f"  [FAIL] {label}  ->  {e}")
        FAILURES.append(label)

print("=" * 60)
print("STAGE 0 VERIFICATION")
print("=" * 60)

print("\n[1/3] Core imports ...")
check("torch",                 lambda: __import__("torch"))
check("transformers",          lambda: __import__("transformers"))
check("datasets",              lambda: __import__("datasets"))
check("flash_attn",            lambda: __import__("flash_attn"))
check("wandb",                 lambda: __import__("wandb"))
check("einops",                lambda: __import__("einops"))
check("tqdm",                  lambda: __import__("tqdm"))
check("numpy",                 lambda: __import__("numpy"))
check("rich",                  lambda: __import__("rich"))
check("safetensors",           lambda: __import__("safetensors"))
check("scipy",                 lambda: __import__("scipy"))
check("rbloom",                lambda: __import__("rbloom"))

print("\n[2/3] Tool-system imports ...")
check("lightrag",              lambda: __import__("lightrag"))
check("pymupdf (fitz)",        lambda: __import__("fitz"))
check("sentence_transformers", lambda: __import__("sentence_transformers"))
check("faiss",                 lambda: __import__("faiss"))

print("\n[3/3] Voice-pipeline imports ...")
check("whisper",               lambda: __import__("whisper"))
check("sounddevice",           lambda: __import__("sounddevice"))
check("soundfile",             lambda: __import__("soundfile"))

print("\n" + "=" * 60)
print("GPU / BF16 checks ...")
import torch

if not torch.cuda.is_available():
    print("  [FAIL] CUDA available  ->  CUDA not found")
    FAILURES.append("CUDA available")
else:
    print("  [OK]  CUDA available")

if not torch.cuda.is_bf16_supported():
    print("  [FAIL] BF16 supported  ->  BF16 not supported")
    FAILURES.append("BF16 supported")
else:
    print("  [OK]  BF16 supported")

try:
    t = torch.zeros(1, dtype=torch.bfloat16).cuda()
    assert t.dtype == torch.bfloat16
    print("  [OK]  BF16 tensor on GPU")
except Exception as e:
    print(f"  [FAIL] BF16 tensor on GPU  ->  {e}")
    FAILURES.append("BF16 tensor on GPU")

if torch.cuda.is_available():
    print(f"\n  GPU : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\n" + "=" * 60)
if FAILURES:
    print(f"STAGE 0 INCOMPLETE -- {len(FAILURES)} failure(s):")
    for f in FAILURES:
        print(f"  * {f}")
    sys.exit(1)
else:
    print("STAGE 0 PASSED")
