from __future__ import annotations

import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F2
import torchvision.transforms.functional as F
from PIL import Image
from tqdm import tqdm

from cxas.augmentation import ControlPointAugmentation
from cxas.checkpoints import load_checkpoint
from cxas.label_mapper import colors
from cxas.models.UNet.unet_model import UNet

THIS_DIR = Path(__file__).parent
DEFAULT_REFERENCE_JSON = THIS_DIR / "data" / "anychest_reference.json"
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".dcm", ".dicom"}


class CXAS_Segmentor:
    def __init__(
        self,
        checkpoint_path: str,
        checkpoint_profile: str | None = None,
        metadata_json_path: str | None = None,
        device: str = "cuda:0",
        save_option: str = "sep",
        selected_views: list[str] | None = None,
        batch_size: int = 8,
        save_format: str = "img",
        num_augmentations: int = 20,
        save_overlay: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.save_format = save_format
        self.num_augmentations = num_augmentations
        self.save_overlay = save_overlay
        self.save_option = save_option
        self.default_aug_params = {
            "base_x_shift": 0.15,
            "base_y_shift": 0.10,
            "shift_range": 0.1,
            "is_extra": False,
        }

        if save_option not in {"one", "sep", "total"}:
            raise ValueError("save_option must be one of: one, sep, total.")
        if save_format not in {"img", "npy"}:
            raise ValueError("save_format must be one of: img, npy.")

        metadata_path = Path(metadata_json_path) if metadata_json_path else DEFAULT_REFERENCE_JSON
        self.reference_config = self._load_reference_json(metadata_path)
        self.view_configs = self.reference_config["view_configs"]
        self.valid_views = set(self.view_configs.keys())
        self.class_names = list(self.reference_config["class_names"])
        self.class_colors = [colors[index + 1][:3] for index in range(len(self.class_names))]

        checkpoint = load_checkpoint(checkpoint_path, profile=checkpoint_profile)
        checkpoint_selected_views = list(checkpoint.get("selected_views") or [])
        self.selected_views = selected_views or checkpoint_selected_views
        invalid_views = [view for view in self.selected_views if view not in self.valid_views]
        if invalid_views:
            raise ValueError(f"Selected views {invalid_views} are not defined in the reference JSON.")
        if not self.selected_views:
            raise ValueError("No active views resolved for inference.")

        self.jtr_start_epoch = checkpoint.get("jtr_start_epoch", math.inf)
        self.base_jtr_c_dist = checkpoint.get("base_jtr_c_dist", 0.0)
        self.base_jtr_c_recon = checkpoint.get("base_jtr_c_recon", 0.0)

        model_name = checkpoint.get("model_name") or "resnet50"
        in_channels = int(checkpoint.get("in_channels", 1))
        n_classes = int(checkpoint.get("n_classes", len(self.class_names)))
        angle_encoding_type = checkpoint.get("angle_encoding_type", "sin_cos")
        jtr_latent_channels = int(checkpoint.get("jtr_latent_channels", 128))
        enable_jtr = self.jtr_start_epoch != math.inf

        self.model = UNet(
            model_name=model_name,
            in_channels=in_channels,
            n_classes=n_classes,
            angle_encoding_type=angle_encoding_type,
            enable_jtr=enable_jtr,
            jtr_latent_channels=jtr_latent_channels,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.target_resolution = checkpoint.get("final_resolution")
        if self.target_resolution is None and checkpoint.get("progressive_training_resolutions"):
            self.target_resolution = checkpoint["progressive_training_resolutions"][-1]
        if isinstance(self.target_resolution, int):
            self.target_resolution = (self.target_resolution, self.target_resolution)
        if self.target_resolution is None:
            self.target_resolution = (512, 512)

        self.mean = self._to_float(checkpoint.get("mean"), default=0.0)
        self.std = self._to_float(checkpoint.get("std"), default=1.0)
        if self.std == 0:
            self.std = 1.0

    @staticmethod
    def _to_float(value: Any, default: float) -> float:
        if value is None:
            return default
        if isinstance(value, torch.Tensor):
            return float(value.item())
        return float(value)

    @staticmethod
    def _load_reference_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _is_supported_file(path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES

    def _load_grayscale_array(self, path: Path) -> np.ndarray:
        suffix = path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg"}:
            return np.array(Image.open(path).convert("L"), dtype=np.float32)
        if suffix in {".dcm", ".dicom"}:
            image = sitk.ReadImage(str(path))
            array = sitk.GetArrayFromImage(image).astype(np.float32)
            if array.ndim == 3:
                array = array[0]
            return array
        raise ValueError(f"Unsupported input file type: {path.suffix}")

    def _load_uint8_overlay_base(self, path: Path) -> np.ndarray:
        array = self._load_grayscale_array(path)
        array = array - array.min()
        max_value = array.max()
        if max_value > 0:
            array = array / max_value
        return (array * 255).astype(np.uint8)

    def _normalize_array(self, array: np.ndarray) -> np.ndarray:
        array = array.astype(np.float32)
        array_min = float(array.min())
        array_max = float(array.max())
        if array_max == array_min:
            return np.zeros_like(array, dtype=np.float32)
        return (array - array_min) / (array_max - array_min)

    def _get_augmentation_params(self, aug_idx: int) -> dict[str, float | bool]:
        if aug_idx == 0:
            return self.default_aug_params
        return {
            "base_x_shift": random.uniform(-0.3, 0.3),
            "base_y_shift": random.uniform(-0.3, 0.3),
            "shift_range": random.uniform(0.0, 0.3),
            "is_extra": False,
        }

    def _preprocess_single(
        self,
        img_path: Path,
        augmentation_params: dict[str, float | bool],
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        original_array = self._load_grayscale_array(img_path)
        original_size = tuple(int(x) for x in original_array.shape)
        image_array = self._normalize_array(original_array)
        image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)

        control_aug = ControlPointAugmentation()
        sample = {"data": image_tensor.unsqueeze(0)}
        augmented_sample = control_aug(sample, augmentation_params)
        image_tensor = augmented_sample["data"].squeeze(0)

        if self.target_resolution:
            image_tensor = F.resize(
                image_tensor,
                self.target_resolution,
                interpolation=F.InterpolationMode.BILINEAR,
            )

        image_tensor = (image_tensor - self.mean) / self.std
        return image_tensor, original_size

    def _preprocess_batch(
        self,
        image_paths: list[Path],
        augmentation_params: dict[str, float | bool],
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        parallel_tasks = [(path, augmentation_params) for path in image_paths]
        with ThreadPoolExecutor(max_workers=min(8, len(parallel_tasks))) as executor:
            results = list(executor.map(lambda item: self._preprocess_single(item[0], item[1]), parallel_tasks))
        batch_tensor = torch.stack([result[0] for result in results])
        original_sizes = [result[1] for result in results]
        return batch_tensor, original_sizes

    def _get_view_angle(self, view_name: str) -> float:
        if view_name not in self.view_configs:
            raise ValueError(f"View '{view_name}' is not defined in the reference JSON.")
        return float(self.view_configs[view_name]["angle"])

    def _save_mask(
        self,
        predicted_mask: np.ndarray,
        img_stem: str,
        output_base_path: Path,
        detected_view_name: str,
        relative_path: Path,
        aug_idx: int | None = None,
    ) -> None:
        rel_dir = relative_path.parent
        base_save_dir = output_base_path / detected_view_name / "labelsTr" / rel_dir
        base_save_dir.mkdir(parents=True, exist_ok=True)

        is_multi_aug_npy = self.save_option == "total" and self.save_format == "npy"
        if is_multi_aug_npy and aug_idx is not None:
            main_folder = base_save_dir / f"{img_stem}_npy"
            save_dir = main_folder / f"{img_stem}_{aug_idx + 1}"
            save_dir.mkdir(parents=True, exist_ok=True)
            for index, class_name in enumerate(self.class_names):
                np.save(save_dir / f"{class_name}.npy", predicted_mask[index])
            return

        if self.save_option == "total":
            save_dir = base_save_dir / f"{img_stem}_total"
            save_dir.mkdir(parents=True, exist_ok=True)
        elif self.save_option == "sep":
            save_dir = base_save_dir / img_stem
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = base_save_dir

        for index, class_name in enumerate(self.class_names):
            if self.save_option == "one":
                file_base = base_save_dir / f"{img_stem}_{class_name}"
            else:
                file_base = save_dir / class_name

            if self.save_format == "img":
                Image.fromarray((predicted_mask[index] * 255).astype(np.uint8)).save(f"{file_base}.png")
            else:
                np.save(f"{file_base}.npy", predicted_mask[index])

    def _save_overlay(
        self,
        source_path: Path,
        predicted_mask: np.ndarray,
        output_base_path: Path,
        detected_view_name: str,
        relative_path: Path,
    ) -> None:
        if not self.save_overlay:
            return

        base_image = self._load_uint8_overlay_base(source_path)
        overlay = np.stack([base_image, base_image, base_image], axis=-1).astype(np.float32)
        for class_index, class_color in enumerate(self.class_colors):
            mask = predicted_mask[class_index] > 0.5
            if not np.any(mask):
                continue
            color_array = np.array(class_color, dtype=np.float32) * 255.0
            overlay[mask] = 0.45 * overlay[mask] + 0.55 * color_array

        rel_dir = relative_path.parent
        overlay_dir = output_base_path / detected_view_name / "overlays" / rel_dir
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = overlay_dir / f"{relative_path.stem}_overlay.png"
        Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(overlay_path)

    def _save_augmented_input_temp_png(
        self,
        input_tensor: np.ndarray,
        img_stem: str,
        output_base_path: Path,
        detected_view_name: str,
        relative_path: Path,
        aug_idx: int,
    ) -> None:
        rel_dir = relative_path.parent
        base_save_dir = output_base_path / detected_view_name / "labelsTr" / rel_dir
        main_folder = base_save_dir / f"{img_stem}_npy" / f"{img_stem}_img"
        main_folder.mkdir(parents=True, exist_ok=True)
        image_array = input_tensor.squeeze()
        if image_array.dtype in {np.float32, np.float64}:
            image_array = (np.clip(image_array, 0.0, 1.0) * 255).astype(np.uint8)
        Image.fromarray(image_array).save(main_folder / f"{img_stem}_{aug_idx + 1}.png")

    def process_batch(
        self,
        batch_data: list[tuple[Path, str, Path]],
        output_base_path: Path,
    ) -> None:
        if not batch_data:
            return

        image_paths = [item[0] for item in batch_data]
        view_names = [item[1] for item in batch_data]
        rel_paths = [item[2] for item in batch_data]
        img_stems = [path.stem for path in image_paths]
        angles = [self._get_view_angle(view_name) for view_name in view_names]
        view_angle_tensor = torch.tensor(angles, dtype=torch.float32, device=self.device)
        is_multi_aug_npy_mode = self.save_option == "total" and self.save_format == "npy"

        if is_multi_aug_npy_mode:
            for aug_idx in range(self.num_augmentations):
                current_aug_params = self._get_augmentation_params(aug_idx)
                batch_tensor, original_sizes = self._preprocess_batch(image_paths, current_aug_params)
                batch_tensor = batch_tensor.to(self.device)

                for index in range(batch_tensor.shape[0]):
                    self._save_augmented_input_temp_png(
                        input_tensor=batch_tensor[index : index + 1].cpu().numpy(),
                        img_stem=img_stems[index],
                        output_base_path=output_base_path,
                        detected_view_name=view_names[index],
                        relative_path=rel_paths[index],
                        aug_idx=aug_idx,
                    )

                with torch.no_grad():
                    outputs = self.model(
                        {
                            "data": batch_tensor,
                            "view_angle": view_angle_tensor,
                            "jtr_c_dist_weight": 0.0,
                            "jtr_c_recon_weight": 0.0,
                        }
                    )
                    logits = outputs["logits"]

                probabilities = torch.sigmoid(logits)
                for index in range(probabilities.shape[0]):
                    prob = probabilities[index : index + 1]
                    original_size = original_sizes[index]
                    if prob.shape[2:] != original_size:
                        prob = F2.interpolate(prob, size=original_size, mode="bilinear", align_corners=True)
                    predicted_mask = (prob > 0.5).float().squeeze(0).cpu().numpy()
                    self._save_mask(
                        predicted_mask=predicted_mask,
                        img_stem=img_stems[index],
                        output_base_path=output_base_path,
                        detected_view_name=view_names[index],
                        relative_path=rel_paths[index],
                        aug_idx=aug_idx,
                    )
            return

        default_aug_params = self._get_augmentation_params(0)
        batch_tensor, original_sizes = self._preprocess_batch(image_paths, default_aug_params)
        batch_tensor = batch_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                {
                    "data": batch_tensor,
                    "view_angle": view_angle_tensor,
                    "jtr_c_dist_weight": 0.0,
                    "jtr_c_recon_weight": 0.0,
                }
            )
            logits = outputs["logits"]

        probabilities = torch.sigmoid(logits)
        for index in range(probabilities.shape[0]):
            prob = probabilities[index : index + 1]
            original_size = original_sizes[index]
            if prob.shape[2:] != original_size:
                prob = F2.interpolate(prob, size=original_size, mode="bilinear", align_corners=True)
            predicted_mask = (prob > 0.5).float().squeeze(0).cpu().numpy()
            self._save_mask(
                predicted_mask=predicted_mask,
                img_stem=img_stems[index],
                output_base_path=output_base_path,
                detected_view_name=view_names[index],
                relative_path=rel_paths[index],
            )
            self._save_overlay(
                source_path=image_paths[index],
                predicted_mask=predicted_mask,
                output_base_path=output_base_path,
                detected_view_name=view_names[index],
                relative_path=rel_paths[index],
            )

    def _infer_view_from_path(self, path: Path) -> str | None:
        parts = [part.lower() for part in path.parts]
        for view_name, view_config in self.view_configs.items():
            folder_name = str(view_config.get("data_folder_name", view_name)).lower()
            if view_name.lower() in parts or folder_name in parts:
                return view_name
        return None

    def _resolve_single_view(self, path: Path, explicit_view_name: str | None) -> str:
        if explicit_view_name:
            if explicit_view_name not in self.valid_views:
                raise ValueError(f"Unknown view name '{explicit_view_name}'.")
            return explicit_view_name

        inferred = self._infer_view_from_path(path)
        if inferred is not None and inferred in self.selected_views:
            return inferred

        if len(self.selected_views) == 1:
            return self.selected_views[0]

        raise ValueError(
            "Could not infer a unique view for the provided input. Please pass --view-name explicitly."
        )

    def _collect_dataset_like_inputs(self, input_base_path: Path) -> list[tuple[Path, str, Path]]:
        all_image_paths: list[tuple[Path, str, Path]] = []
        for view_name in self.selected_views:
            view_folder_name = self.view_configs[view_name].get("data_folder_name", view_name)
            base_images_path = input_base_path / view_folder_name / "imagesTr"
            if not base_images_path.is_dir():
                continue
            files = [
                path
                for path in base_images_path.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            ]
            for file_path in files:
                all_image_paths.append((file_path, view_name, file_path.relative_to(base_images_path)))
        return sorted(all_image_paths, key=lambda item: str(item[0]))

    def _collect_flat_inputs(
        self,
        input_base_path: Path,
        explicit_view_name: str | None,
    ) -> list[tuple[Path, str, Path]]:
        files = [
            path
            for path in input_base_path.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ]
        if not files:
            return []
        view_name = self._resolve_single_view(input_base_path, explicit_view_name)
        return [(file_path, view_name, file_path.relative_to(input_base_path)) for file_path in sorted(files)]

    def process_path(
        self,
        input_path: str,
        output_path: str,
        view_name: str | None = None,
    ) -> None:
        input_base_path = Path(input_path)
        output_base_path = Path(output_path)
        output_base_path.mkdir(parents=True, exist_ok=True)

        if input_base_path.is_file():
            if not self._is_supported_file(input_base_path):
                raise ValueError(f"Unsupported input file type: {input_base_path.suffix}")
            single_view = self._resolve_single_view(input_base_path, view_name)
            all_image_paths = [(input_base_path, single_view, Path(input_base_path.name))]
        elif input_base_path.is_dir():
            all_image_paths = self._collect_dataset_like_inputs(input_base_path)
            if not all_image_paths:
                all_image_paths = self._collect_flat_inputs(input_base_path, view_name)
        else:
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        if not all_image_paths:
            raise ValueError(f"No supported image files were found under: {input_path}")

        total_batches = (len(all_image_paths) + self.batch_size - 1) // self.batch_size
        for batch_idx in tqdm(range(total_batches), desc="Inference"):
            start = batch_idx * self.batch_size
            end = min((batch_idx + 1) * self.batch_size, len(all_image_paths))
            self.process_batch(all_image_paths[start:end], output_base_path)

    def process_folder(self, input_folder_path: str, output_folder_path: str) -> None:
        self.process_path(input_folder_path, output_folder_path)
