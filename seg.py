from __future__ import annotations

import argparse
import os
from pathlib import Path

from cxas.checkpoints import AVAILABLE_BUNDLE_PROFILES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AnyChest inference on a single image, a flat folder, or a dataset-style folder tree."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to a PNG, JPEG, DICOM file, or to a folder containing supported files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for predicted masks and overlays.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a local AnyChest checkpoint file or merged inference bundle downloaded from Hugging Face.",
    )
    parser.add_argument(
        "--profile",
        choices=AVAILABLE_BUNDLE_PROFILES,
        default=None,
        help="Bundle profile to load when --checkpoint points to the merged HF bundle.",
    )
    parser.add_argument(
        "--view-name",
        default=None,
        help="Explicit view name for flat folders or single-file inference. Required for ambiguous oblique inputs.",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="Optional path to a custom inference reference JSON. Defaults to the packaged AnyChest reference.",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size.")
    parser.add_argument(
        "--save-option",
        choices=["one", "sep", "total"],
        default="sep",
        help="Mask save mode.",
    )
    parser.add_argument(
        "--save-format",
        choices=["img", "npy"],
        default="img",
        help="Mask output format.",
    )
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=20,
        help="Number of inference augmentations used only for --save-option total --save-format npy.",
    )
    parser.add_argument(
        "--disable-overlays",
        action="store_true",
        help="Skip color overlay export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from cxas.segmentor import CXAS_Segmentor

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    segmentor = CXAS_Segmentor(
        checkpoint_path=str(args.checkpoint),
        checkpoint_profile=args.profile,
        metadata_json_path=str(args.metadata_json) if args.metadata_json else None,
        device=args.device,
        save_option=args.save_option,
        batch_size=args.batch_size,
        save_format=args.save_format,
        num_augmentations=args.num_augmentations,
        save_overlay=not args.disable_overlays,
    )
    segmentor.process_path(
        input_path=str(args.input_path),
        output_path=str(args.output_dir),
        view_name=args.view_name,
    )


if __name__ == "__main__":
    main()
