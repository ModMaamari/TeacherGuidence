"""Simple GPU access test.

Checks GPU access in the order the layers actually stack, and stops at the first one that
fails -- so the output points at the real cause instead of the symptom:

    1. nvidia-smi          driver talks to the cards at all
    2. driver versions     loaded kernel module vs userspace libraries (must match)
    3. CUDA init           cudaGetDeviceCount succeeds
    4. per-GPU compute     actually allocate memory and run a matmul on each card

A card can look perfectly healthy in ``nvidia-smi`` and still be unusable for CUDA, so
step 4 is the one that decides whether a training or vLLM run will work.

Usage::

    .venv_vllm/bin/python scripts/check_gpu.py
    .venv_vllm/bin/python scripts/check_gpu.py --gpu 1      # test a single card
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys


def sh(cmd: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return p.returncode, (p.stdout + p.stderr).strip()
    except FileNotFoundError:
        return 127, f"{cmd[0]}: not found"
    except Exception as exc:  # noqa: BLE001
        return 1, f"{type(exc).__name__}: {exc}"


def check_nvidia_smi() -> bool:
    print("1. nvidia-smi")
    rc, out = sh(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                  "--format=csv,noheader,nounits"])
    if rc != 0:
        print(f"   FAIL: {out[:200]}")
        return False
    for line in out.splitlines():
        idx, name, used, total, util = [x.strip() for x in line.split(",")]
        print(f"   GPU {idx}: {name:24} {used:>6}/{total} MiB  util {util}%")
    return True


def check_driver_versions() -> bool:
    """The loaded kernel module and the userspace libraries must be the same version.

    After a driver package upgrade without a reboot they differ, and every CUDA call
    fails with a confusing error while nvidia-smi keeps working.
    """
    print("2. driver versions")
    kernel = None
    try:
        with open("/proc/driver/nvidia/version") as fh:
            text = fh.read()
        match = re.search(r"Kernel Module\s+([\d.]+)", text)
        kernel = match.group(1) if match else None
        print(f"   kernel module : {kernel or 'unknown'}")
    except OSError as exc:
        print(f"   kernel module : unreadable ({exc})")

    rc, out = sh(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    userspace = out.splitlines()[0].strip() if rc == 0 and out else None
    print(f"   nvidia-smi    : {userspace or 'unknown'}")

    if kernel and userspace and kernel != userspace:
        print(f"   FAIL: version mismatch ({kernel} vs {userspace})")
        print("         the driver was upgraded but the running kernel module was not "
              "reloaded -- reboot, or reload the nvidia modules")
        return False
    print("   OK: versions match" if kernel and userspace else "   (could not compare)")
    return True


def check_cuda_init() -> bool:
    print("3. CUDA initialization")
    try:
        import torch
    except ImportError:
        print("   FAIL: torch is not installed in this interpreter")
        print(f"         try: .venv_vllm/bin/python {sys.argv[0]}")
        return False

    print(f"   torch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"   OK: {torch.cuda.device_count()} device(s) visible")
        return True

    # is_available() swallows the real error; provoke it to report something actionable.
    try:
        torch.cuda.init()
        detail = "unknown"
    except Exception as exc:  # noqa: BLE001
        detail = str(exc).split("\n")[0]
    print(f"   FAIL: torch.cuda.is_available() is False")
    print(f"         {detail[:200]}")
    if "system not yet initialized" in detail or "802" in detail:
        print("         error 802 on A100-SXM/NVSwitch hosts almost always means the")
        print("         Fabric Manager is not running or does not match the driver:")
        print("           systemctl status nvidia-fabricmanager")
        print("           systemctl start  nvidia-fabricmanager")
    return False


def check_compute(only: int | None) -> bool:
    print("4. per-GPU compute")
    import torch

    count = torch.cuda.device_count()
    targets = [only] if only is not None else list(range(count))
    all_ok = True
    for idx in targets:
        if idx >= count:
            print(f"   GPU {idx}: FAIL - only {count} device(s) visible")
            all_ok = False
            continue
        try:
            device = torch.device(f"cuda:{idx}")
            a = torch.randn(2048, 2048, device=device, dtype=torch.float32)
            result = float((a @ a).sum())
            free, total = torch.cuda.mem_get_info(device)
            name = torch.cuda.get_device_name(device)
            print(f"   GPU {idx}: OK   {name:24} matmul ok, "
                  f"{free / 2**30:.1f}/{total / 2**30:.1f} GiB free")
            del a
            torch.cuda.empty_cache()
            if result != result:  # NaN
                print(f"   GPU {idx}: WARN matmul produced NaN")
        except Exception as exc:  # noqa: BLE001
            print(f"   GPU {idx}: FAIL {type(exc).__name__}: {str(exc).splitlines()[0][:150]}")
            all_ok = False
    return all_ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", type=int, default=None, help="test only this GPU index")
    args = ap.parse_args()

    print("=== GPU access check ===\n")
    steps = [
        ("nvidia-smi", check_nvidia_smi),
        ("driver versions", check_driver_versions),
        ("CUDA init", check_cuda_init),
    ]
    for name, step in steps:
        if not step():
            print(f"\n=== BLOCKED at: {name} — GPUs are NOT usable ===")
            sys.exit(1)
        print()

    ok = check_compute(args.gpu)
    print()
    if ok:
        print("=== ALL CHECKS PASSED — GPUs are usable ===")
    else:
        print("=== SOME GPUs FAILED — see above ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
