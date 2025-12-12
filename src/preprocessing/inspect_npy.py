import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


def iter_npy_files(root: Path, categories: Iterable[str]) -> Iterable[Tuple[str, Path]]:
    """
    Iterează prin fișierele .npy din categoriile specificate.
    """
    for cat in categories:
        for npy in sorted((root / cat).glob("*_imu.npy")):
            yield cat, npy


def summarize_array(arr: np.ndarray) -> dict:
    """
    Generează un sumar statistic pentru un array numpy.
    """
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
    """
    Inspectează fișierele dintr-un folder și afișează statistici de bază.
    """
    total = 0
    per_cat = {}
    for cat, npy_path in iter_npy_files(root, categories):
        try:
            data = np.load(npy_path)
        except Exception as e:
            print(f"Eroare: {npy_path}: {e}")
            continue

        s = summarize_array(data)
        print(f"{cat}: {npy_path.name} -> formă={s['shape']} tip={s['dtype']} nan={s['nan']}" +
              (f" min={s.get('min')} max={s.get('max')} medie={s.get('mean')} std={s.get('std')}" if 'min' in s else ""))

        total += 1
        per_cat[cat] = per_cat.get(cat, 0) + 1
        if limit and total >= limit:
            break

    print("\nRezumat:")
    print(f"  Total fișiere: {total}")
    for cat in categories:
        print(f"  {cat}: {per_cat.get(cat, 0)}")


def main():
    parser = argparse.ArgumentParser(description="Inspectare fișiere NPY")
    parser.add_argument("--root", default=str(Path("data") / "raw"))
    parser.add_argument("--categories", nargs="*", default=["asphalt", "carpet", "concrete", "grass", "tile"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    inspect_folder(Path(args.root), args.categories, args.limit)


if __name__ == "__main__":
    main()
