# cxas/dataset.py
import os
import glob
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from sklearn.model_selection import KFold
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import torchvision.transforms.functional as F
from cxas.augmentation import DeterministicAugmentor, AugmentationPlanner
import random
import re


class MultiChannelSegDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        file_ext: str = ".png",
        is_train: bool = True,
        epoch: int = 0,
        augmentation_plan: Optional[AugmentationPlanner] = None,
        selected_views: Optional[List[str]] = None,
        selected_classes: Optional[List[str]] = None,
        target_resolution: Optional[Tuple[int, int]] = None,
        mean: float = 0.5,        # 新增：标准化均值
        std: float = 0.5,          # 新增：标准化标准差
        normalize: bool = True,      # 新增：是否进行标准化
        use_variations: bool = False,  # 新增参数：是否使用增强变体 
        eval_model: bool = True
    ):
        """Initialize dataset with paths and configurations.
        Args:
            base_dir: Root directory containing the dataset.json and view-specific data folders.
            file_ext: File extension for image and mask files (default '.png').
            is_train: Whether this is training data (affects augmentation).
            epoch: Current epoch for deterministic augmentation.
            augmentation_plan: Pre-generated augmentation plan dictionary.
            selected_views: Optional list of view names (e.g., ['AP', 'PA']) to include.
                            If None, all views defined in dataset.json will be used.
            selected_classes: Optional list of class names (e.g., ['aorta', 'heart']) to include.
                              If None, all classes defined in global_class_weights will be used.
            target_resolution: Optional tuple (H, W) for resizing images and masks. If None, original resolution is kept.
        """
        self.rng = random.Random()
        self.rng.seed(42) 
        self.base_dir = base_dir 
        self.json_path = Path(base_dir) / "dataset.json"
        self.file_ext = file_ext
        self.is_train = is_train
        self.epoch = epoch
        self.target_resolution = target_resolution
        self.mean = mean          # 新增：保存标准化参数
        self.std = std            # 新增：保存标准化参数
        self.normalize = normalize  # 新增：保存标准化参数
        self.use_variations = use_variations  # 保存开关状态
        self.eval_model = eval_model  # 保存评估模式状态

        

        self.augmentor = None
        if is_train:
            if augmentation_plan is not None:
                # 直接使用AugmentationPlanner实例初始化增强器
                self.augmentor = DeterministicAugmentor(augmentation_plan, epoch)
                print(f"MultiChannelSegDataset: 已初始化增强器（epoch={epoch}）")
            else:
                print("Warning: 训练集未提供增强计划，将不应用确定性增强")

        self.dataset_config = self._load_dataset_json()
        
        all_view_names = list(self.dataset_config.get("view_configs", {}).keys())
        if selected_views:
            invalid_views = [v for v in selected_views if v not in all_view_names]
            if invalid_views:
                raise ValueError(f"Selected views {invalid_views} are not defined in dataset.json view_configs.")
            self.active_views_info = {v_name: self.dataset_config["view_configs"][v_name] for v_name in selected_views}
        else:
            self.active_views_info = self.dataset_config.get("view_configs", {}) 

        self.label_info = self._create_label_info(selected_classes=selected_classes)
        self.ids = self._collect_ids() 
            
        

        if is_train and self.use_variations:  # 仅在训练模式且启用变体时调用
            self.sample_to_variations = self._build_variation_map()
        else:
            self.sample_to_variations = {}  # 不使用变体时设置为空字典

    def _load_dataset_json(self) -> Dict[str, Any]:
        """Load and parse the dataset JSON file."""
        with open(self.json_path, 'r') as f:
            return json.load(f)

    def _create_label_info(self, selected_classes: Optional[List[str]] = None) -> Dict[str, Any]:
        """Dynamically create class mappings and collect weights from 'global_class_weights'.
        Filters classes based on selected_classes and assigns integer IDs based on insertion order.
        """
        id_to_name = {}
        name_to_id = {}
        name_to_mask_filename = {}
        raw_class_weights_by_id = {}

        global_class_weights_dict = self.dataset_config.get("global_class_weights", {})
        
        current_id = 0 

        if "background" in global_class_weights_dict:
            name_to_id["background"] = 0
            id_to_name[0] = "background"
            raw_class_weights_by_id[0] = global_class_weights_dict["background"]
            name_to_mask_filename["background"] = "" 
            current_id = 1 

        for class_name, weight in global_class_weights_dict.items():
            if class_name == "background":
                continue 

            if selected_classes is not None and class_name not in selected_classes:
                continue 

            _id = current_id
            id_to_name[_id] = class_name
            name_to_id[class_name] = _id
            name_to_mask_filename[class_name] = f"{class_name}{self.file_ext}"
            raw_class_weights_by_id[_id] = weight
            current_id += 1

        max_id = max(raw_class_weights_by_id.keys()) if raw_class_weights_by_id else -1
        class_weights_list = [0.0] * (max_id + 1) if max_id >= 0 else []
        for _id, weight in raw_class_weights_by_id.items():
            class_weights_list[_id] = weight

        sorted_class_ids = sorted([_id for _id in id_to_name.keys() if id_to_name[_id] != "background"])

        return {
            "id_to_name": id_to_name,
            "name_to_id": name_to_id,
            "name_to_mask_filename": name_to_mask_filename,
            "class_weights": class_weights_list, 
            "sorted_class_ids": sorted_class_ids, 
            "num_output_channels": len(sorted_class_ids) 
        }

    def _collect_ids(self) -> List[Tuple[str, str]]:
        """
        Collects (image_id, view_name) tuples from all active view directories.
        Ensures a matching '_total' suffixed label subfolder exists for each image.
        Prints warnings for mismatches or missing directories.
        """
        all_ids_with_views = []
        for view_name, view_info in self.active_views_info.items():
            view_data_folder = view_info["data_folder_name"]
            current_image_dir = os.path.join(self.base_dir, view_data_folder, 'imagesTr')
            current_label_dir = os.path.join(self.base_dir, view_data_folder, 'labelsTr')
            
            # Initial checks for existence of base directories
            if not os.path.exists(current_image_dir):
                print(f"Warning: Dataset: Image directory not found for view '{view_name}': {current_image_dir}. Skipping.")
                continue
            if not os.path.exists(current_label_dir):
                print(f"Warning: Dataset: Label directory not found for view '{view_name}': {current_label_dir}. Skipping.")
                continue

            image_files_paths = glob.glob(os.path.join(current_image_dir, f"*{self.file_ext}"))
            image_base_ids = {os.path.splitext(os.path.basename(f))[0] for f in image_files_paths}
            
            # 如果启用了 use_variations，则只加载图像本体
            if not self.use_variations and not self.eval_model:
                # 使用正则表达式匹配图像本体的命名格式
                pattern = re.compile(r"^(t|train)_[a-zA-Z0-9]+_[a-zA-Z0-9]+_[a-zA-Z0-9]+$")
                image_base_ids = {img_id for img_id in image_base_ids if pattern.match(img_id)}

            
            label_subfolder_paths = glob.glob(os.path.join(current_label_dir, '*'))
            label_subfolders = {os.path.basename(d) for d in label_subfolder_paths if os.path.isdir(d)}

            # Warn if counts don't match (before filtering for _total suffix)
            if len(image_base_ids) != len(label_subfolders):
                print(f"Warning: Dataset: Image count ({len(image_base_ids)}) does not match label subfolder count ({len(label_subfolders)}) for view '{view_name}'. This may indicate missing data.")

            processed_count = 0
            skipped_count = 0

            for base_id_image in image_base_ids:
                label_folder_name_expected = f"{base_id_image}_total" 

                if label_folder_name_expected in label_subfolders:
                    all_ids_with_views.append((base_id_image, view_name))
                    processed_count += 1
                else:
                    print(f"Warning: Dataset: No matching label folder '{label_folder_name_expected}' found for image '{base_id_image}' in '{current_label_dir}'. Skipping image.")
                    skipped_count += 1
            
            if skipped_count > 0:
                print(f"Summary: Dataset: For view '{view_name}', {processed_count} images processed, {skipped_count} images skipped due to missing label folders.")

        if not all_ids_with_views:
            raise ValueError(f"No valid image-label pairs found across all selected views. Please check data paths and naming conventions.")

        print(f"Info: Dataset: Successfully collected {len(all_ids_with_views)} image-label pairs across all views.")
        return sorted(all_ids_with_views)

    def __len__(self) -> int:
        """Get total number of samples in dataset."""
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load and preprocess a single sample by index, including view angle and relevant class weights."""
        try:
            base_id_image, view_name = self.ids[idx] 
            view_info = self.active_views_info[view_name]
            view_angle = view_info["angle"]
            data_folder_name = view_info["data_folder_name"]

            original_image_path = os.path.join(
                self.base_dir, data_folder_name, 'imagesTr', f"{base_id_image}{self.file_ext}"
            )
            candidate_image_paths = [original_image_path]
            # 如果启用了变体并且是训练模式，才添加变体路径
            if self.is_train and self.use_variations:
                    if base_id_image in self.sample_to_variations:
                        candidate_image_paths.extend(self.sample_to_variations[base_id_image])

            # 3. 随机选择一个图像路径（原始或增强变体）
            selected_image_path = self.rng.choice(candidate_image_paths)

            # 读取图像
            image = Image.open(selected_image_path).convert("L")
            image_array = np.array(image, dtype=np.float32)
            
            # 自适应归一化处理
            img_min = np.min(image_array)
            img_max = np.max(image_array)
            
            # 避免除零错误
            if img_max == img_min:
                img_max += 1e-8
                img_min += 1e-8
                
            if self.normalize:    
                image_array = (image_array - img_min) / (img_max - img_min)
            image_tensor = torch.tensor(image_array).unsqueeze(0) 

            if self.target_resolution:
                image_tensor = F.resize(image_tensor, self.target_resolution, interpolation=F.InterpolationMode.BILINEAR)

            mask_tensors = []
            has_foreground_pixels = False # Flag to check for foreground pixels in this sample

            active_class_weights = torch.tensor(
                [self.label_info["class_weights"][_id] for _id in self.label_info["sorted_class_ids"]],
                dtype=torch.float32
            )

            mask_tensors = []
            has_foreground_pixels = False

            label_folder_for_image = f"{base_id_image}_total"  # 标签文件夹命名规则
            label_base_dir_for_view_and_id = os.path.join(
                self.base_dir, data_folder_name, 'labelsTr', label_folder_for_image
            )


            for i, class_id in enumerate(self.label_info["sorted_class_ids"]):
                class_name = self.label_info["id_to_name"][class_id]
                mask_filename = self.label_info["name_to_mask_filename"][class_name]
                
                mask_path = os.path.join(label_base_dir_for_view_and_id, mask_filename)
                
                file_exists = os.path.exists(mask_path)

                if file_exists:
                    mask = Image.open(mask_path).convert("L")
                    mask_array = np.array(mask)
                    class_mask = (mask_array > 0).astype(np.float32) 
                    if np.any(class_mask > 0): # Check if this specific mask has foreground
                        has_foreground_pixels = True
                else:
                    target_h, target_w = self.target_resolution if self.target_resolution else (image_array.shape[0], image_array.shape[1])
                    class_mask = np.zeros((target_h, target_w), dtype=np.float32) 
                    active_class_weights[i] = 0.0

                class_mask_tensor = torch.tensor(class_mask).unsqueeze(0) 

                if self.target_resolution and class_mask_tensor.shape[2:] != self.target_resolution:
                     class_mask_tensor = F.resize(class_mask_tensor, self.target_resolution, interpolation=F.InterpolationMode.NEAREST)
                mask_tensors.append(class_mask_tensor.squeeze(0)) 
                
            label_tensor = torch.stack(mask_tensors) 

            # NEW: Warning for entirely black samples
            if not has_foreground_pixels and self.is_train: # Only warn for training samples
                 print(f"Warning: Dataset: Sample '{base_id_image}' ({view_name}) has no foreground labels (all black masks) after loading all classes. This sample might not contribute positively to training.")
            
            
            sample_mask_weight_sum = active_class_weights.sum().item()
            sample = {
                "id": base_id_image, 
                "data": image_tensor,
                "label": label_tensor,
                "class_weights": active_class_weights,
                "view_name": view_name, 
                "view_angle": torch.tensor(float(view_angle), dtype=torch.float32) ,
                "sample_global_weight": sample_mask_weight_sum
            }
            sample["data"] = sample["data"].contiguous()
            sample["label"] = sample["label"].contiguous()
            
            if self.augmentor:
                sample = self.augmentor(sample)

            if self.normalize:
                sample["data"] = (sample["data"] - self.mean) / self.std

            return sample
        except Exception as e:
            raise RuntimeError(f"Error loading sample {idx} (ID: {self.ids[idx] if idx < len(self.ids) else 'N/A'}): {str(e)}")


    def _build_variation_map(self):
        # 遍历所有样本，构建 {样本ID: [变体路径列表]} 的映射
        variation_map = {}
        for view_name, view_info in self.active_views_info.items():
            flavor_dir = os.path.join(self.base_dir, view_info["data_folder_name"], "flavor")
            if os.path.isdir(flavor_dir):
                for sample_id in os.listdir(flavor_dir):
                    sample_flavor_dir = os.path.join(flavor_dir, sample_id)
                    if os.path.isdir(sample_flavor_dir):
                        variations = glob.glob(os.path.join(sample_flavor_dir, f"*{self.file_ext}"))
                        if variations:
                            variation_map[sample_id] = variations
        return variation_map



def get_5fold_dataloaders(
    base_dir: str,
    fold_id: int = 0,
    file_ext: str = ".png",
    batch_size: int = 4,
    num_workers: int = 8,
    augmentation_planner: Optional[Any] = None,
    selected_views: Optional[List[str]] = None,
    selected_classes: Optional[List[str]] = None,
    current_resolution: Optional[Tuple[int, int]] = None,
    mean: float = 0.5,
    std: float = 0.5,
    use_variations: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """简化版：仅LA视图带flavor的样本优先划分到训练集，其他情况随机划分"""

    # 加载数据集配置，确定视图选择（原有逻辑不变）
    json_path = Path(base_dir) / "dataset.json"
    if not json_path.exists():
        raise FileNotFoundError(f"dataset.json not found at {json_path}")
    
    with open(json_path, 'r') as f:
        dataset_config = json.load(f)

    dataset_mode_config = dataset_config.get("dataset_mode", {})
    multi_view_enabled = dataset_mode_config.get("multi_view_enabled", True) 
    
    final_selected_views = []
    if multi_view_enabled:
        if selected_views:
            final_selected_views = selected_views
        else:
            final_selected_views = list(dataset_config.get("view_configs", {}).keys())
        if not final_selected_views:
            raise ValueError("Multi-view mode is enabled but no views are selected or defined in dataset.json.")
    else:
        default_view_name = dataset_mode_config.get("default_view")
        if default_view_name and default_view_name in dataset_config.get("view_configs", {}):
            final_selected_views = [default_view_name]
        else:
            raise ValueError("Single-view mode is enabled but 'default_view' is not specified or not found in 'view_configs'.")

    # 初始化数据集（原有逻辑不变）
    train_dataset = MultiChannelSegDataset(
        base_dir,
        file_ext=file_ext,
        is_train=True,
        epoch=0, 
        augmentation_plan=augmentation_planner,
        selected_views=final_selected_views, 
        selected_classes=selected_classes, 
        target_resolution=current_resolution,
        mean=mean,
        std=std,
        use_variations=use_variations  # 传递开关参数
    )
    val_dataset = MultiChannelSegDataset(
        base_dir,
        file_ext=file_ext,
        is_train=False, 
        selected_views=final_selected_views, 
        selected_classes=selected_classes, 
        target_resolution=current_resolution,
        mean=mean,
        std=std
    )

    total_samples = len(train_dataset)
    all_sample_info = train_dataset.ids  # 格式: [(base_id, view_name), ...]

    # --------------------------
    # 简化逻辑：仅标记LA视图带flavor的样本为优先级
    # --------------------------
    priority_indices = []  # 优先放入训练集的样本（LA+flavor）
    other_indices = []     # 其他样本（非LA，或LA但无flavor）

    # 仅当选中LA视图时，才筛选优先级样本
    if "LA" in final_selected_views:
        for idx, (base_id, view_name) in enumerate(all_sample_info):
            # 只关注LA视图，且存在flavor变体的样本
            if view_name == "LA" and base_id in train_dataset.sample_to_variations:
                priority_indices.append(idx)
            else:
                other_indices.append(idx)
        print(f"LA视图带flavor的优先级样本: {len(priority_indices)} 个")
    else:
        # 未选中LA视图，所有样本均为普通样本
        other_indices = list(range(total_samples))
        print("未选中LA视图，所有样本随机划分")

    # --------------------------
    # 划分逻辑：优先将LA+flavor样本放入训练集
    # --------------------------
    train_ratio = 0.8
    train_size = int(total_samples * train_ratio)
    val_size = total_samples - train_size

    # 训练集基础：尽可能包含所有优先级样本（LA+flavor）
    # 若优先级样本数超过训练集大小，则随机选部分
    priority_train = priority_indices
    # 剩余训练名额从其他样本中随机补充
    remaining_train_slots = train_size - len(priority_train)


    # 构建5折划分（保持原有逻辑不变）
    kf_splits = []
    for fold in range(5):
        # 基础训练集：固定优先级样本
        fold_priority_train = priority_train.copy()
        
        # 剩余名额随机选择（每折用不同种子保证差异）
        np.random.seed(42 + fold)  # 按fold偏移种子
        remaining_train_slots = train_size - len(fold_priority_train)
        fold_remaining_train = np.random.choice(
            other_indices, size=remaining_train_slots, replace=False
        ).tolist() if remaining_train_slots > 0 else []

        # 训练集索引 = 优先级样本 + 随机补充样本
        fold_train_idx = fold_priority_train + fold_remaining_train
        # 验证集索引 = 总样本 - 训练集索引
        fold_val_idx = [idx for idx in range(total_samples) if idx not in fold_train_idx]

        # 校验大小
        assert len(fold_train_idx) == train_size, f"fold {fold} 训练集大小错误"
        assert len(fold_val_idx) == val_size, f"fold {fold} 验证集大小错误"
        kf_splits.append((fold_train_idx, fold_val_idx))

    # 获取当前fold的划分
    train_idx, val_idx = kf_splits[fold_id]

    # 创建DataLoader
    train_loader = DataLoader(
        Subset(train_dataset, indices=train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=custom_collate
    )

    val_loader = DataLoader(
        Subset(val_dataset, indices=val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=custom_collate
    )

    return train_loader, val_loader


def custom_collate(batch):
    # 仅保留张量字段，拼接为 batch
    batch_data = {
        "data": torch.stack([sample["data"] for sample in batch]),
        "label": torch.stack([sample["label"] for sample in batch]),
        "class_weights": torch.stack([sample["class_weights"] for sample in batch]),
        "view_angle": torch.stack([sample["view_angle"] for sample in batch])
    }
    # 非张量字段单独存储（不参与 pin_memory）
    batch_meta = {
        "id": [sample["id"] for sample in batch],
        "view_name": [sample["view_name"] for sample in batch],
        "sample_global_weight": [sample["sample_global_weight"] for sample in batch]
    }
    # 将元数据附加到 batch 中（模型不需要，仅用于日志/后处理）
    batch_data["meta"] = batch_meta
    return batch_data

def get_model_channels(dataset_json_path: str, selected_classes: Optional[List[str]] = None) -> Tuple[int, int]:
    """Get input/output channel counts for model configuration based on selected classes.
    Args:
        dataset_json_path: Path to dataset.json file.
        selected_classes: Optional list of class names to include. If None, all classes from global_class_weights (excluding background) are counted.
    Returns:
        Tuple of (input_channels, output_channels).
    Note:
        Assumes grayscale input (1 channel) and one output channel per selected class.
    """
    import json 
    
    with open(dataset_json_path, "r") as f:
        dataset_config = json.load(f)
    
    global_class_weights = dataset_config.get("global_class_weights", {})
    
    if not global_class_weights:
        raise ValueError("dataset.json must contain 'global_class_weights' key with class definitions.")

    active_classes = []
    for class_name in global_class_weights.keys():
        if class_name == "background":
            continue 
        if selected_classes is None or class_name in selected_classes:
            active_classes.append(class_name)

    num_output_channels = len(active_classes)
        
    return 1, num_output_channels