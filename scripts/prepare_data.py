from __future__ import annotations

import tarfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    archive_path = root / "dataset" / "food-101.tar.gz"
    target_root = root / "data" / "raw"
    expected_dir = target_root / "food-101"
    target_root.mkdir(parents=True, exist_ok=True)
    if expected_dir.exists():
        print(f"Dataset already prepared at {expected_dir}")
        return
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(target_root)
    print(f"Extracted dataset to {expected_dir}")


if __name__ == "__main__":
    main()

