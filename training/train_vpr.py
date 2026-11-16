"""Train one outer fold of the PRINT VPR inference module using inner cross-validation."""
import argparse
import os
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "mae", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--workbook", type=Path)
    source.add_argument("--splits", type=Path)
    p.add_argument("--targets", type=Path, help="RPR CSV when targets are supplied separately")
    p.add_argument("--fold", required=True, type=int, choices=range(10))
    p.add_argument("--epochs", default=150, type=int, choices=range(1, 151), metavar="1..150",
                   help="Maximum number of epochs per inner fold")
    p.add_argument("--steps", default=0, type=int, help="Stop after this many optimizer updates and save a checkpoint (0 runs the full schedule)")
    a = p.parse_args()
    if a.splits and not a.targets:
        p.error("--splits requires --targets")
    if a.steps < 0:
        p.error("--steps must be nonnegative")
    os.environ["THYROID_DATA_ROOT"] = str(a.data.resolve())
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    runtime = Path(__file__).resolve().parents[1] / "print_model/runtime"
    sys.path[:0] = list(map(str, (runtime/"scripts", runtime/"third_party/URFM", runtime)))
    sys.path.insert(0, str(runtime.parents[1]))
    from vpr_training_engine import train
    from nested_cv import run
    run(a, train)

if __name__ == "__main__":
    main()
