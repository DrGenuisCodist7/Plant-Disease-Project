"""Prepare leakage-aware Tomato PlantVillage train/validation/test folders.

Run from the project folder:
    python prepare_tomato_dataset.py

The script copies (never moves) images from PlantVillage-Dataset-master/raw/color,
uses leaf-map.json to keep photographs of the same physical leaf together, and
creates 70/15/15 train/val/test folders plus reproducibility manifests.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def normalized_image_identifier(filename: str) -> str:
    """Match the identifier-building logic used by PlantVillage metadata."""
    identifier = Path(filename).stem
    identifier = identifier.replace("_final_masked", "")
    if "___" in identifier:
        identifier = identifier.split("___")[-1]
    identifier = identifier.lower().split("copy")[0].strip()
    return identifier


def physical_leaf_id(class_name: str, filename: str, leaf_map: dict) -> tuple[str, bool]:
    identifier = normalized_image_identifier(filename)
    suggestions = leaf_map.get(identifier, [])
    if len(suggestions) == 1:
        return str(suggestions[0]), True
    for suggestion in suggestions:
        if class_name in str(suggestion):
            return str(suggestion), True
    # The fallback still groups files whose names differ only by a "copy" suffix.
    return f"{class_name}:::fallback::{identifier}", False


def allocate_groups(groups: dict[str, list[Path]], seed: int) -> dict[str, str]:
    """Allocate whole leaf groups while approximately matching image ratios."""
    rng = random.Random(seed)
    items = list(groups.items())
    rng.shuffle(items)
    # Larger groups first reduces ratio error; the random value resolves equal sizes.
    items.sort(key=lambda item: len(item[1]), reverse=True)

    total = sum(len(paths) for _, paths in items)
    targets = {name: total * ratio for name, ratio in SPLIT_RATIOS.items()}
    counts = {name: 0 for name in SPLIT_RATIOS}
    assignment = {}

    for leaf_id, paths in items:
        # Select the split with the largest remaining proportional need.
        split = max(
            SPLIT_RATIOS,
            key=lambda name: (targets[name] - counts[name]) / max(targets[name], 1),
        )
        assignment[leaf_id] = split
        counts[split] += len(paths)
    return assignment


def ensure_empty_destination(destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit(
            f"STOP: '{destination}' is not empty. Rename or remove that prepared dataset "
            "only after checking it, then run this script again."
        )
    destination.mkdir(parents=True, exist_ok=True)


def prepare(args) -> None:
    source = Path(args.source).resolve()
    map_path = Path(args.leaf_map).resolve()
    destination = Path(args.output).resolve()

    if not source.is_dir():
        raise SystemExit(f"Source folder not found: {source}")
    if not map_path.is_file():
        raise SystemExit(f"Leaf map not found: {map_path}")

    class_dirs = sorted(p for p in source.iterdir() if p.is_dir() and p.name.startswith("Tomato___"))
    if len(class_dirs) != 10:
        raise SystemExit(f"Expected 10 tomato classes, but found {len(class_dirs)} in {source}")

    leaf_map = json.loads(map_path.read_text(encoding="utf-8"))
    if not isinstance(leaf_map, dict):
        raise SystemExit("leaf-map.json does not contain the expected dictionary structure")

    records = []
    mapped_images = fallback_images = 0
    for class_dir in class_dirs:
        images = sorted(
            p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        groups = defaultdict(list)
        mapping_flags = {}
        for image_path in images:
            leaf_id, mapped = physical_leaf_id(class_dir.name, image_path.name, leaf_map)
            groups[leaf_id].append(image_path)
            mapping_flags[leaf_id] = mapped
            mapped_images += int(mapped)
            fallback_images += int(not mapped)

        assignment = allocate_groups(groups, args.seed)
        for leaf_id, grouped_images in groups.items():
            split = assignment[leaf_id]
            for image_path in grouped_images:
                records.append(
                    {
                        "source": image_path,
                        "class": class_dir.name,
                        "split": split,
                        "leaf_id": leaf_id,
                        "leaf_map_match": mapping_flags[leaf_id],
                    }
                )

    # Final safety check: no physical leaf may occur in more than one split.
    leaf_splits = defaultdict(set)
    for record in records:
        leaf_splits[record["leaf_id"]].add(record["split"])
    leakage = {leaf: splits for leaf, splits in leaf_splits.items() if len(splits) > 1}
    if leakage:
        raise SystemExit(f"Safety check failed: {len(leakage)} leaf groups cross splits")

    ensure_empty_destination(destination)
    for split in SPLIT_RATIOS:
        for class_dir in class_dirs:
            (destination / split / class_dir.name).mkdir(parents=True, exist_ok=True)

    for record in records:
        target = destination / record["split"] / record["class"] / record["source"].name
        shutil.copy2(record["source"], target)
        record["target"] = target

    manifest = destination / "split_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "class", "filename", "leaf_id", "leaf_map_match"])
        for record in sorted(records, key=lambda r: (r["split"], r["class"], r["source"].name)):
            writer.writerow(
                [record["split"], record["class"], record["source"].name,
                 record["leaf_id"], record["leaf_map_match"]]
            )

    summary = {
        "seed": args.seed,
        "ratios": SPLIT_RATIOS,
        "classes": [p.name for p in class_dirs],
        "total_images": len(records),
        "mapped_images": mapped_images,
        "fallback_images": fallback_images,
        "images_per_split": {
            split: sum(r["split"] == split for r in records) for split in SPLIT_RATIOS
        },
        "leaf_groups_per_split": {
            split: sum(split in splits for splits in leaf_splits.values()) for split in SPLIT_RATIOS
        },
        "leakage_groups": 0,
    }
    (destination / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Tomato dataset prepared successfully")
    print(f"Classes: {len(class_dirs)}")
    print(f"Total images: {len(records)}")
    for split, count in summary["images_per_split"].items():
        print(f"{split:>5}: {count} images")
    print(f"Leaf-map matched images: {mapped_images}")
    print(f"Fallback-grouped images: {fallback_images}")
    print("Leakage check: PASS (0 leaf groups cross splits)")
    print(f"Manifest: {manifest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=r"PlantVillage-Dataset-master\raw\color",
        help="Folder containing the 38 PlantVillage colour class folders",
    )
    parser.add_argument(
        "--leaf-map",
        default=r"PlantVillage-Dataset-master\leaf-map.json",
        help="PlantVillage physical-leaf mapping JSON",
    )
    parser.add_argument("--output", default="dataset", help="New prepared dataset folder")
    parser.add_argument("--seed", type=int, default=42, help="Reproducible split seed")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
