"""Run every Lab 1 step in order. From repo root: python lab01/src/run_all.py"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
RESULTS = SRC.parent / "results"


def run(script, tee=None):
    print(f"\n### {script}")
    out = subprocess.run([sys.executable, str(SRC / script)], cwd=SRC,
                         capture_output=True, text=True, check=True).stdout
    print(out)
    if tee:
        (RESULTS / tee).write_text(out)


if __name__ == "__main__":
    run("print_versions.py", tee="versions.txt")
    run("train_baselines.py")
    run("measure_cost.py")
    run("budget_table.py")
