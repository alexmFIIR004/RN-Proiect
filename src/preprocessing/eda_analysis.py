import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np


def load_category_data(category_dir: Path) -> List[np.ndarray]:
    arrays = []
    for npy_file in sorted(category_dir.glob("*_imu.npy")):
        try:
            data = np.load(npy_file)
            arrays.append(data)
        except Exception as e:
            print(f"Warning: {npy_file}: {e}")
    return arrays


def compute_stats(arrays: List[np.ndarray]) -> Dict:
    if not arrays:
        return {}
    
    stacked = np.stack(arrays, axis=0)
    flat = stacked.reshape(-1, stacked.shape[-1])
    
    stats = {
        "n_samples": len(arrays),
        "shape_per_sample": list(arrays[0].shape),
        "dtype": str(arrays[0].dtype),
        "mean": np.mean(flat, axis=0).tolist(),
        "median": np.median(flat, axis=0).tolist(),
        "std": np.std(flat, axis=0).tolist(),
        "min": np.min(flat, axis=0).tolist(),
        "max": np.max(flat, axis=0).tolist(),
        "q25": np.percentile(flat, 25, axis=0).tolist(),
        "q75": np.percentile(flat, 75, axis=0).tolist(),
        "nan_count": int(np.isnan(flat).sum()),
        "inf_count": int(np.isinf(flat).sum()),
    }
    
    q1 = np.percentile(flat, 25, axis=0)
    q3 = np.percentile(flat, 75, axis=0)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers_per_feature = []
    for col in range(flat.shape[1]):
        outliers = ((flat[:, col] < lower_bound[col]) | (flat[:, col] > upper_bound[col])).sum()
        outliers_per_feature.append(int(outliers))
    
    stats["outliers_per_feature"] = outliers_per_feature
    stats["total_outliers"] = sum(outliers_per_feature)
    
    return stats


def analyze_dataset(raw_root: Path, categories: List[str]) -> Dict:
    results = {"categories": {}}
    
    for cat in categories:
        cat_dir = raw_root / cat
        if not cat_dir.exists():
            print(f"Warning: {cat_dir} not found")
            continue
        
        print(f"Analyzing: {cat}")
        arrays = load_category_data(cat_dir)
        stats = compute_stats(arrays)
        results["categories"][cat] = stats
    
    counts = {cat: results["categories"][cat].get("n_samples", 0) for cat in categories}
    results["class_balance"] = counts
    results["total_samples"] = sum(counts.values())
    
    return results


def print_summary(results: Dict):
    print("\n" + "="*60)
    print("EDA SUMMARY")
    print("="*60)
    
    print(f"\nTotal samples: {results['total_samples']}")
    print("\nClass balance:")
    for cat, count in results["class_balance"].items():
        print(f"  {cat}: {count}")
    
    print("\n" + "-"*60)
    print("Per-category statistics:")
    print("-"*60)
    
    feature_names = [
        "orientation_x", "orientation_y", "orientation_z", "orientation_w",
        "angular_vel_x", "angular_vel_y", "angular_vel_z",
        "linear_acc_x", "linear_acc_y", "linear_acc_z"
    ]
    
    for cat, stats in results["categories"].items():
        print(f"\n{cat.upper()}:")
        print(f"  Samples: {stats.get('n_samples', 0)}")
        print(f"  Shape per sample: {stats.get('shape_per_sample', [])}")
        print(f"  NaN count: {stats.get('nan_count', 0)}")
        print(f"  Inf count: {stats.get('inf_count', 0)}")
        print(f"  Total outliers (IQR): {stats.get('total_outliers', 0)}")
        
        if "mean" in stats:
            print(f"\n  Feature-wise statistics:")
            for i, fname in enumerate(feature_names):
                print(f"    {fname}:")
                print(f"      mean={stats['mean'][i]:.4f}, median={stats['median'][i]:.4f}, std={stats['std'][i]:.4f}")
                print(f"      min={stats['min'][i]:.4f}, max={stats['max'][i]:.4f}")
                print(f"      Q1={stats['q25'][i]:.4f}, Q3={stats['q75'][i]:.4f}")
                print(f"      outliers={stats['outliers_per_feature'][i]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", default="data/raw")
    parser.add_argument("--categories", nargs="*", default=["asphalt", "carpet", "concrete", "grass", "tile"])
    parser.add_argument("--output", default="docs/datasets/eda_results.json")
    args = parser.parse_args()
    
    raw_root = Path(args.raw_root)
    results = analyze_dataset(raw_root, args.categories)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    print_summary(results)


if __name__ == "__main__":
    main()
