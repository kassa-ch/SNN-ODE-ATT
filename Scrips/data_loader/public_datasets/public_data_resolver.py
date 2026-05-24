import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_ROOT = PROJECT_ROOT / "Results" / "manifests" / "splits"


@dataclass(frozen=True)
class SplitRatio:
    train: float = 0.70
    val: float = 0.15
    test_normal: float = 0.15

    def validate(self):
        total = self.train + self.val + self.test_normal
        if any(x < 0 for x in (self.train, self.val, self.test_normal)):
            raise ValueError("Split ratios must be non-negative.")
        if total <= 0:
            raise ValueError("At least one split ratio must be positive.")
        return total


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except Exception:
        return str(path)


def find_public_label_paths(data_dir):
    data_dir = Path(data_dir)
    normal = sorted(data_dir.glob("*_normal.csv"))
    abnormal = sorted(data_dir.glob("*_abnormal.csv"))
    return normal, abnormal


def _split_normal_paths(paths, ratio: SplitRatio, seed: int):
    ratio_total = ratio.validate()
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    train_n = int(n * ratio.train / ratio_total)
    val_n = int(n * ratio.val / ratio_total)

    train = shuffled[:train_n]
    val = shuffled[train_n:train_n + val_n]
    test_normal = shuffled[train_n + val_n:]
    return train, val, test_normal


def resolve_public_split(
    dataset,
    view,
    data_dir,
    seed=42,
    train_ratio=0.70,
    val_ratio=0.15,
    test_normal_ratio=0.15,
    save=True,
):
    data_dir = Path(data_dir)
    normal, abnormal = find_public_label_paths(data_dir)
    ratio = SplitRatio(train=train_ratio, val=val_ratio, test_normal=test_normal_ratio)
    train_normal, val_normal, test_normal = _split_normal_paths(normal, ratio, seed)

    split = {
        "dataset": dataset,
        "view": view,
        "data_dir": _rel(data_dir),
        "seed": seed,
        "policy": {
            "train": "normal only",
            "val": "normal only",
            "test": "normal + abnormal",
            "label_patterns": {
                "normal": "*_normal.csv",
                "abnormal": "*_abnormal.csv",
            },
        },
        "counts": {
            "normal_total": len(normal),
            "abnormal_total": len(abnormal),
            "train_normal": len(train_normal),
            "val_normal": len(val_normal),
            "test_normal": len(test_normal),
            "test_abnormal": len(abnormal),
        },
        "train_normal_paths": [_rel(p) for p in train_normal],
        "val_normal_paths": [_rel(p) for p in val_normal],
        "test_normal_paths": [_rel(p) for p in test_normal],
        "test_abnormal_paths": [_rel(p) for p in abnormal],
    }

    if save:
        MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
        out = MANIFEST_ROOT / f"{dataset.lower().replace('-', '_')}_{view}_split_seed{seed}.json"
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(split, handle, indent=2, ensure_ascii=False)
        split["manifest_path"] = _rel(out)

    return split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--view", default="uniform", choices=["uniform", "nonuniform"])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-normal-ratio", type=float, default=0.15)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    split = resolve_public_split(
        dataset=args.dataset,
        view=args.view,
        data_dir=args.data_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_normal_ratio=args.test_normal_ratio,
        save=not args.no_save,
    )
    print(json.dumps({"counts": split["counts"], "manifest_path": split.get("manifest_path")}, indent=2))


if __name__ == "__main__":
    main()
