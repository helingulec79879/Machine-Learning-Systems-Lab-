"""Print the Python version and the version of every package pinned in requirements.txt.

Usage (from repo root):
    python lab01/src/print_versions.py | tee lab01/results/versions.txt
"""
import platform
import sys
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REQ_FILE = REPO_ROOT / "requirements.txt"


def pinned_packages():
    pkgs = []
    for line in REQ_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, pinned = line.partition("==")
        pkgs.append((name.strip(), pinned.strip()))
    return pkgs


def main():
    print(f"Python      : {sys.version.split()[0]}")
    print(f"Executable  : {sys.executable}")
    print(f"Platform    : {platform.platform()}")
    print(f"Machine     : {platform.machine()} | CPU: {platform.processor() or 'n/a'}")
    print()
    print(f"{'package':<22}{'installed':<28}{'pinned':<28}status")
    print("-" * 86)
    mismatches = 0
    for name, pinned in pinned_packages():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = "NOT INSTALLED"
        status = "OK" if installed == pinned else "MISMATCH"
        if status != "OK":
            mismatches += 1
        print(f"{name:<22}{installed:<28}{pinned:<28}{status}")
    print("-" * 86)
    print(f"{mismatches} package(s) differ from requirements.txt")
    if not sys.version.startswith("3.11.8"):
        print("WARNING: required Python is 3.11.8")


if __name__ == "__main__":
    main()
