import argparse
import shutil
from pathlib import Path
from typing import List, Tuple
import random
import numpy as np
from preprocess_images import preprocess_image, save_processed_image
from preprocess_imu import preprocess_imu, save_processed_imu


def find_pairs(category_dir: Path) -> List[Tuple[Path, Path, str]]:
    pairs = []
    imgs = sorted(category_dir.glob("*_img.jpg"))
    for img in imgs:
        stem = img.name.replace("_img.jpg", "")
        imu = category_dir / f"{stem}_imu.npy"
        if imu.exists():
            pairs.append((img, imu, stem))
    return pairs


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def split_indices(n: int, train_ratio: float, val_ratio: float) -> Tuple[List[int], List[int], List[int]]:
    train_n = int(n * train_ratio)
    val_n = int(n * val_ratio)
    indices = list(range(n))
    random.shuffle(indices)
    return indices[:train_n], indices[train_n:train_n+val_n], indices[train_n+val_n:]


def copy_pair(img: Path, imu: Path, out_dir: Path, stem: str, category: str, stats: dict = None):
    ensure_dir(out_dir)
    dark_min = stats.get('dark_min', 0) if stats else 0
    light_max = stats.get('light_max', 255) if stats else 255
    
    processed_img = preprocess_image(img, target_size=(224, 224), dark_min=dark_min, light_max=light_max)
    if processed_img is not None:
        save_processed_image(processed_img, out_dir / f"{category}_{stem}_img.jpg")
    else:
        shutil.copy2(img, out_dir / f"{category}_{stem}_img.jpg")

    imu_data = preprocess_imu(str(imu))
    if imu_data is not None:
        save_processed_imu(imu_data, out_dir / f"{category}_{stem}_imu.npy")
    else:
        shutil.copy2(imu, out_dir / f"{category}_{stem}_imu.npy")


def compute_train_stats(pairs: List[Tuple[Path, Path, str]], train_idx: List[int]) -> dict:
    all_pixels = []
    for i in train_idx[:50]:
        img_path = pairs[i][0]
        try:
            import cv2
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                all_pixels.extend(img.flatten())
        except:
            pass
    if all_pixels:
        pixels = np.array(all_pixels)
        return {'dark_min': int(np.percentile(pixels, 1)), 'light_max': int(np.percentile(pixels, 99))}
    return {'dark_min': 0, 'light_max': 255}


def process_category(category: str, raw_root: Path, processed_root: Path, splits_root: Path, train_ratio: float, val_ratio: float):
    cat_dir = raw_root / category
    pairs = find_pairs(cat_dir)
    if not pairs:
        print(f"Warning: no pairs found '{category}'")
        return

    n = len(pairs)
    train_idx, val_idx, test_idx = split_indices(n, train_ratio, val_ratio)
    
    stats = compute_train_stats(pairs, train_idx)
    
    processed_cat = processed_root / category
    ensure_dir(processed_cat)
    for img, imu, stem in pairs:
        copy_pair(img, imu, processed_cat, stem, category, stats)

    index_lists = [(train_idx, splits_root / "train" / category),
                   (val_idx, splits_root / "validation" / category),
                   (test_idx, splits_root / "test" / category)]

    for idx_list, out_dir in index_lists:
        ensure_dir(out_dir)
        for i in idx_list:
            img, imu, stem = pairs[i]
            copy_pair(img, imu, out_dir, stem, category, stats)


def main():
    parser = argparse.ArgumentParser()
    
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    
    parser.add_argument("--raw_root", default=str(project_root / "data" / "raw"))
    parser.add_argument("--processed_root", default=str(project_root / "data" / "processed"))
    parser.add_argument("--splits_root", default=str(project_root / "data"))
    parser.add_argument("--categories", nargs="*", default=["asphalt", "carpet", "concrete", "grass", "tile"])
    parser.add_argument("--train_ratio", type=float, default=0.75)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    raw_root = Path(args.raw_root)
    processed_root = Path(args.processed_root)
    splits_root = Path(args.splits_root)

    ensure_dir(processed_root)
    for sub in [splits_root / "train", splits_root / "validation", splits_root / "test"]:
        ensure_dir(sub)

    for category in args.categories:
        process_category(category, raw_root, processed_root, splits_root, args.train_ratio, args.val_ratio)

    print("Done.")


if __name__ == "__main__":
    main()
