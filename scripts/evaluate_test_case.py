from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cxas.segmentor import CXAS_Segmentor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AnyCXR inference on a Dataset003-style test_case folder and compute Dice summaries."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path to Dataset003_Full/test_case.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the merged HF bundle or local checkpoint.")
    parser.add_argument("--profile", required=True, help="Bundle profile, e.g. la, pa, oblique, oblique_45.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for predictions and reports.")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size.")
    parser.add_argument("--skip-inference", action="store_true", help="Only score an existing prediction directory.")
    return parser.parse_args()


def load_mask(path: Path) -> np.ndarray:
    return (np.array(Image.open(path), dtype=np.uint8) > 127).astype(np.uint8)


def calc_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred_bin = pred_mask.astype(bool)
    gt_bin = gt_mask.astype(bool)
    total = pred_bin.sum() + gt_bin.sum()
    if total == 0:
        return 1.0
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    return float((2.0 * intersection) / total)


def collect_metrics(
    dataset_dir: Path,
    pred_dir: Path,
    view_names: list[str],
    class_names: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    per_image_rows: list[dict[str, Any]] = []
    all_scores: dict[tuple[str, str], list[float]] = {}

    for view_name in view_names:
        data_folder_name = {
            "22.5 Degree": "225",
            "Oblique 45 Degree": "45",
            "67.5 Degree": "675",
            "112.5 Degree": "1125",
            "Oblique 135 Degree": "135",
            "157.5 Degree": "1575",
        }.get(view_name, view_name)

        image_dir = dataset_dir / data_folder_name / "imagesTr"
        gt_dir = dataset_dir / data_folder_name / "labelsTr"
        pred_view_dir = pred_dir / view_name / "labelsTr"
        if not image_dir.is_dir() or not gt_dir.is_dir() or not pred_view_dir.is_dir():
            continue

        for image_path in sorted(image_dir.glob("*.png")):
            sample_id = image_path.stem
            gt_sample_dir = gt_dir / f"{sample_id}_total"
            pred_sample_dir = pred_view_dir / sample_id
            if not gt_sample_dir.is_dir() or not pred_sample_dir.is_dir():
                continue

            image_scores: list[float] = []
            for class_name in class_names:
                gt_mask_path = gt_sample_dir / f"{class_name}.png"
                pred_mask_path = pred_sample_dir / f"{class_name}.png"
                if not gt_mask_path.is_file() or not pred_mask_path.is_file():
                    continue
                score = calc_dice(load_mask(pred_mask_path), load_mask(gt_mask_path))
                per_image_rows.append(
                    {
                        "view_name": view_name,
                        "sample_id": sample_id,
                        "class_name": class_name,
                        "dice": score,
                    }
                )
                all_scores.setdefault((view_name, class_name), []).append(score)
                image_scores.append(score)

            if image_scores:
                per_image_rows.append(
                    {
                        "view_name": view_name,
                        "sample_id": sample_id,
                        "class_name": "__mean__",
                        "dice": float(np.mean(image_scores)),
                    }
                )

    per_class_rows: list[dict[str, Any]] = []
    for (view_name, class_name), scores in sorted(all_scores.items()):
        per_class_rows.append(
            {
                "view_name": view_name,
                "class_name": class_name,
                "dice_mean": float(np.mean(scores)),
                "dice_std": float(np.std(scores)),
                "n_samples": len(scores),
            }
        )

    summary = {
        "views": view_names,
        "n_class_entries": len(per_class_rows),
        "overall_mean_dice": float(np.mean([row["dice_mean"] for row in per_class_rows])) if per_class_rows else None,
    }
    return per_image_rows, per_class_rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    report_dir = args.output_dir / "reports"

    segmentor = CXAS_Segmentor(
        checkpoint_path=str(args.checkpoint),
        checkpoint_profile=args.profile,
        device=args.device,
        batch_size=args.batch_size,
    )

    if not args.skip_inference:
        segmentor.process_path(str(args.dataset_dir), str(args.output_dir))

    per_image_rows, per_class_rows, summary = collect_metrics(
        dataset_dir=args.dataset_dir,
        pred_dir=args.output_dir,
        view_names=segmentor.selected_views,
        class_names=segmentor.class_names,
    )

    write_csv(
        report_dir / "per_image_dice.csv",
        per_image_rows,
        ["view_name", "sample_id", "class_name", "dice"],
    )
    write_csv(
        report_dir / "per_class_summary.csv",
        per_class_rows,
        ["view_name", "class_name", "dice_mean", "dice_std", "n_samples"],
    )
    with (report_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
