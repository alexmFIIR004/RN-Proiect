import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


def iter_npy_files(root: Path, categories: Iterable[str]) -> Iterable[Tuple[str, Path]]:
    for cat in categories:
        for npy in sorted((root / cat).glob("*_imu.npy")):
            yield cat, npy


def summarize_array(arr: np.ndarray) -> dict:
    summary = {
        "dtype": str(arr.dtype),
        "shape": tuple(arr.shape),
        "nan": int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0,
    }
    if arr.ndim >= 1 and np.issubdtype(arr.dtype, np.floating):
        flat = arr.reshape(-1)
        summary.update({
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
        })
    return summary


def inspect_folder(root: Path, categories: Iterable[str], limit: int | None = None):
    total = 0
    per_cat = {}
    for cat, npy_path in iter_npy_files(root, categories):
        try:
            data = np.load(npy_path)
        except Exception as e:
            print(f"Error: {npy_path}: {e}")
            continue

        s = summarize_array(data)
        print(f"{cat}: {npy_path.name} -> shape={s['shape']} dtype={s['dtype']} nan={s['nan']}" +
              (f" min={s.get('min')} max={s.get('max')} mean={s.get('mean')} std={s.get('std')}" if 'min' in s else ""))

        total += 1
        per_cat[cat] = per_cat.get(cat, 0) + 1
        if limit and total >= limit:
            break

    print("\nSummary:")
    print(f"  Total files: {total}")
    for cat in categories:
        print(f"  {cat}: {per_cat.get(cat, 0)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path("data") / "raw"))
    parser.add_argument("--categories", nargs="*", default=["asphalt", "carpet", "concrete", "grass", "tile"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    inspect_folder(Path(args.root), args.categories, args.limit)


if __name__ == "__main__":
    main()
