"""Resolve numbered study inputs and existing cohort filenames."""
from pathlib import Path

def study_input(root: Path, name: str) -> Path:
    path = root / name
    if path.exists():
        return path
    for prefix, previous in (("stage1_", "stageB_"), ("stage2_", "stageA_")):
        if name.startswith(prefix):
            existing = root / (previous + name[len(prefix):])
            if existing.exists():
                return existing
    return path
