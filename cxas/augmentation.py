# cxas/augmentation.py
import os
import glob
import torch
import random
import pickle
import json # Import json to read dataset.json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import torchvision.transforms.functional as F
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from scipy.ndimage import gaussian_filter
import cv2
import numpy as np

# (AugmentationType and AugmentationConfig classes remain the same)
class AugmentationType(Enum):
    FLIP_HORIZONTAL = auto()
    FLIP_VERTICAL = auto()
    ROTATE = auto()
    BRIGHTNESS = auto()
    CONTRAST = auto()
    GAMMA = auto()
    NOISE = auto()
    ELASTIC = auto()
    CONTROL_POINT = auto()
    RandomScatterArtifact = auto()
    RandomGlobalContrastStretch = auto()
    RandomGrayDrift = auto()

@dataclass
class AugmentationConfig:
    type: AugmentationType
    probability: float
    params: Optional[Dict[str, Any]] = None
    
    
class AugmentationPlanner:
    """Plans all augmentations for all epochs and images ahead of time"""
    def __init__(self, 
                 base_dir: str,
                 num_epochs: int,
                 augmentation_configs: List[AugmentationConfig],
                 selected_views: Optional[List[str]] = None): 
        self.base_dir = base_dir
        self.num_epochs = num_epochs
        self.configs = augmentation_configs
        self.plan_path = Path(base_dir) / "augmentation_plan.pkl"
        self.selected_views = selected_views

        dataset_json_path = Path(base_dir) / "dataset.json"
        if not dataset_json_path.exists():
            raise FileNotFoundError(f"dataset.json not found at {dataset_json_path}")
        
        with open(dataset_json_path, 'r') as f:
            dataset_config = json.load(f)

        all_view_names = list(dataset_config.get("view_configs", {}).keys())
        if self.selected_views:
            invalid_views = [v for v in self.selected_views if v not in all_view_names]
            if invalid_views:
                raise ValueError(f"Selected views {invalid_views} for AugmentationPlanner are not defined in dataset.json view_configs.")
            self.active_views_info = {v_name: dataset_config["view_configs"][v_name] for v_name in self.selected_views}
        else:
            self.active_views_info = dataset_config.get("view_configs", {}) 

        # Get all image IDs (now view-aware and correctly matching label folders)
        self.image_ids = self._collect_image_ids() # Call new method
        
        self.seed_map = self._generate_or_load_seed_map()  # 新增：存储种子而非全量参数


    
    # MODIFIED: _collect_image_ids to match label folder naming convention
    def _collect_image_ids(self) -> List[Tuple[str, str]]:
        """
        Collects (image_id, view_name) tuples from all active view directories.
        Ensures that a corresponding label subfolder with '_total' suffix exists.
        """
        all_ids_with_views = []
        for view_name, view_info in self.active_views_info.items():
            view_data_folder = view_info["data_folder_name"]
            current_image_dir = os.path.join(self.base_dir, view_data_folder, 'imagesTr')
            current_label_dir = os.path.join(self.base_dir, view_data_folder, 'labelsTr')
            
            print(f"AugmentationPlanner: Checking image directory: {current_image_dir}")
            print(f"AugmentationPlanner: Checking label directory: {current_label_dir}")
            
            if not os.path.exists(current_image_dir):
                print(f"Warning: AugmentationPlanner: Image directory not found for view '{view_name}': {current_image_dir}. Skipping.")
                continue
            if not os.path.exists(current_label_dir):
                print(f"Warning: AugmentationPlanner: Label directory not found for view '{view_name}': {current_label_dir}. Skipping.")
                continue

            image_files = glob.glob(os.path.join(current_image_dir, "*.png")) 
            print(f"AugmentationPlanner: Found {len(image_files)} image files in {current_image_dir} for view '{view_name}'.")
            
            # Collect available label subfolders for validation/matching
            label_subfolders = {os.path.basename(d) for d in glob.glob(os.path.join(current_label_dir, '*')) if os.path.isdir(d)}
            print(f"AugmentationPlanner: Found {len(label_subfolders)} label subfolders in {current_label_dir}.")

            for f_path in image_files:
                base_id_image = os.path.splitext(os.path.basename(f_path))[0]
                
                # Assume label folder name is base_id_image + "_total"
                label_folder_name = f"{base_id_image}_total" 

                # Only add to image_ids if a matching label folder exists
                if label_folder_name in label_subfolders:
                    all_ids_with_views.append((base_id_image, view_name)) 
                else:
                    print(f"Warning: AugmentationPlanner: No matching label folder '{label_folder_name}' found for image '{base_id_image}' in '{current_label_dir}'. Skipping image for planning.")

        print(f"AugmentationPlanner: Total IDs collected across all views: {len(all_ids_with_views)}")
        return sorted(all_ids_with_views)

    def _generate_or_load_seed_map(self) -> Dict[Tuple[int, str, str], int]:
        seed_path = Path(self.base_dir) / "augmentation_seeds.pkl"
        if seed_path.exists():
            with open(seed_path, 'rb') as f:
                print(f"AugmentationPlanner: 已加载种子文件：{seed_path}")
                return pickle.load(f)
        # 为每个(epoch, img_id, view_name)生成唯一种子（保证复现性）
        seed_map = {}
        for epoch in range(self.num_epochs):
            for img_id, view_name in self.image_ids:
                key = (epoch, img_id, view_name)
                seed_map[key] = random.getrandbits(32)  # 32位随机种子
        with open(seed_path, 'wb') as f:
            pickle.dump(seed_map, f)
        print(f"AugmentationPlanner: 已生成新种子文件：{seed_path}（共{len(seed_map)}条记录）")

        return seed_map

    def get_augmentations(self, epoch: int, img_id: str, view_name: str) -> List[Dict[str, Any]]:
        key = (epoch, img_id, view_name)
        if key not in self.seed_map:
            raise ValueError(f"Seed not found: epoch={epoch}, img_id={img_id}, view_name={view_name}")
        
        # 保存并恢复随机状态，避免全局污染
        original_random_state = random.getstate()
        try:
            random.seed(self.seed_map[key])  # 固定种子
            img_augs = []
            for config in self.configs:
                if config.type == AugmentationType.CONTROL_POINT:
                    base_params = self._get_params_for_aug(config)
                    is_extra = random.random() < config.probability
                    base_params["is_extra"] = is_extra
                    img_augs.append({"type": config.type.name, "params": base_params})
                else:
                    if random.random() < config.probability:
                        img_augs.append({
                            "type": config.type.name,
                            "params": self._get_params_for_aug(config)
                        })
            return img_augs
        finally:
            random.setstate(original_random_state)  # 恢复原始状态
    

    
    def _get_params_for_aug(self, config: AugmentationConfig) -> Dict[str, Any]:
        params_dict = config.params if config.params is not None else {}

        if config.type == AugmentationType.ROTATE:
            return {"angle": random.uniform(-params_dict.get("max_angle", 15), 
                                            params_dict.get("max_angle", 15))}
        elif config.type in [AugmentationType.BRIGHTNESS, 
                            AugmentationType.CONTRAST, 
                            AugmentationType.GAMMA]:
            return {"factor": 1 + random.uniform(-params_dict.get("range", 0.1), 
                                                params_dict.get("range", 0.1))}
        elif config.type == AugmentationType.NOISE:
            return {"std": random.uniform(0, params_dict.get("max_std", 0.05))}
        elif config.type in [AugmentationType.FLIP_HORIZONTAL, AugmentationType.FLIP_VERTICAL]:
            return {}
        elif config.type == AugmentationType.CONTROL_POINT:
            return {
                "base_x_shift": params_dict.get("base_x_shift", 0.15),
                "base_y_shift": params_dict.get("base_y_shift", 0.10),
                "shift_range": params_dict.get("shift_range", 0.1)
            }
        # 添加新的增强类型参数配置
        elif config.type == AugmentationType.RandomScatterArtifact:
            return {
                "intensity_range": params_dict.get("intensity_range", [5, 30]),
                "count_range": params_dict.get("count_range", [10, 50])
            }
        elif config.type == AugmentationType.RandomGlobalContrastStretch:
            return {
                "contrast_range": params_dict.get("contrast_range", [0.5, 2.0])
            }
        elif config.type == AugmentationType.RandomGrayDrift:
            return {
                "drift_range": params_dict.get("drift_range", [-20, 20])
            }
        else:
            return {}
        
class DeterministicAugmentor:
    """Applies augmentations according to pre-generated plan"""
    def __init__(self, planner: AugmentationPlanner, epoch: int):  # 接收planner而非plan
        self.planner = planner  # 保存planner实例
        self.epoch = epoch  # 当前epoch
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        img_id = sample["id"]
        view_name = sample["view_name"]
        # 动态获取参数（替代原从plan中查询）
        augmentations = self.planner.get_augmentations(self.epoch, img_id, view_name)
        
        # 应用增强（与原逻辑完全一致，无需修改）
        for aug in augmentations:
            sample = self._apply_augmentation(sample, aug)
        return sample
    
    def _apply_augmentation(self, sample: dict, aug: dict) -> dict:
        aug_type = AugmentationType[aug["type"]]
        params = aug.get("params", {})
        
        if aug_type == AugmentationType.FLIP_HORIZONTAL:
            sample["data"] = F.hflip(sample["data"])
            sample["label"] = F.hflip(sample["label"])
        elif aug_type == AugmentationType.FLIP_VERTICAL:
            sample["data"] = F.vflip(sample["data"])
            sample["label"] = F.vflip(sample["label"])
        elif aug_type == AugmentationType.ROTATE:
            angle = params.get("angle", 0)
            sample["data"] = F.rotate(sample["data"], angle)
            sample["label"] = F.rotate(sample["label"], angle)
        elif aug_type == AugmentationType.NOISE:
            std = params.get("std", 0.02)
            noise = torch.randn_like(sample["data"]) * std
            sample["data"] = sample["data"].to(dtype=torch.float32)  # 统一类型
            sample["data"] = torch.clamp(sample["data"] + noise, 0.0, 1.0)  # 截断范围
        elif aug_type == AugmentationType.CONTROL_POINT:
            # 直接传入预生成的参数（x_shift和y_shift），无需基础参数
            control_point_aug = ControlPointAugmentation()
            sample = control_point_aug(sample, params)  # 将计划生成的params传入
        elif aug_type == AugmentationType.BRIGHTNESS:
            factor = params.get("factor", 1.0)
            # 限制因子范围（避免极端值）
            factor = max(0.5, min(1.5, factor))  # 例如限制在[0.5, 1.5]
            sample["data"] = F.adjust_brightness(sample["data"], factor)
            sample["data"] = torch.clamp(sample["data"], 0.0, 1.0)  # 截断到[0,1]

        elif aug_type == AugmentationType.CONTRAST:
            factor = params.get("factor", 1.0)
            factor = max(0.5, min(1.5, factor))
            sample["data"] = F.adjust_contrast(sample["data"], factor)
            sample["data"] = torch.clamp(sample["data"], 0.0, 1.0)

        elif aug_type == AugmentationType.GAMMA:
            gamma = params.get("factor", 1.0)
            gamma = max(0.8, min(1.2, gamma))  # 伽马因子更严格，避免过曝/过暗
            sample["data"] = F.adjust_gamma(sample["data"], gamma)
            sample["data"] = torch.clamp(sample["data"], 0.0, 1.0)
        elif aug_type == AugmentationType.RandomScatterArtifact:
            scatter_aug = RandomScatterArtifact(
                intensity_range=params.get("intensity_range", [5, 30]),
                count_range=params.get("count_range", [10, 50])
            )
            sample = scatter_aug(sample)
        elif aug_type == AugmentationType.RandomGlobalContrastStretch:
            contrast_aug = RandomGlobalContrastStretch(
                contrast_range=params.get("contrast_range", [0.5, 2.0])
            )
            sample = contrast_aug(sample)
        elif aug_type == AugmentationType.RandomGrayDrift:
            gray_aug = RandomGrayDrift(
                drift_range=params.get("drift_range", [-20, 20])
            )
            sample = gray_aug(sample)
    
        
        sample["data"] = sample["data"].to(dtype=torch.float32)
        return sample

class RandomFlip:
    """Random horizontal and vertical flip"""
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() > 0.5:
            sample["data"] = F.hflip(sample["data"])
            sample["label"] = F.hflip(sample["label"])
        if random.random() > 0.5:
            sample["data"] = F.vflip(sample["data"])
            sample["label"] = F.vflip(sample["label"])
        return sample

class RandomRotate:
    """Random rotation within specified angle range"""
    def __init__(self, max_angle: int = 15):
        self.max_angle = max_angle
        
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        angle = random.uniform(-self.max_angle, self.max_angle)
        sample["data"] = F.rotate(sample["data"], angle)
        sample["label"] = F.rotate(sample["label"], angle)
        return sample


class ControlPointAugmentation:
    """
    基于控制点优化的图像增强器（改进版）
    所有图片必用基础偏移，抽选图片额外增加偏移
    """
    def __init__(self):
        pass
    
    def apply_control_point_adjustment(self, 
                                      image: torch.Tensor, 
                                      original_min: torch.Tensor, 
                                      original_max: torch.Tensor,
                                      new_control_x: torch.Tensor,
                                      new_control_y: float) -> torch.Tensor:
        """保持原逻辑不变，核心控制点调整实现"""
        x_points = torch.tensor([original_min, new_control_x, original_max], device=image.device)
        y_points = torch.tensor([0.0, new_control_y, 1.0], device=image.device)
        
        adjusted_image = torch.zeros_like(image)
        mask_lower = image <= new_control_x
        slope_lower = (new_control_y - 0.0) / (new_control_x - original_min + 1e-8)  # 防除零
        adjusted_image[mask_lower] = slope_lower * (image[mask_lower] - original_min)
        
        mask_upper = image > new_control_x
        slope_upper = (1.0 - new_control_y) / (original_max - new_control_x + 1e-8)
        adjusted_image[mask_upper] = new_control_y + slope_upper * (image[mask_upper] - new_control_x)
        
        return torch.clamp(adjusted_image, 0, 1).contiguous()
    
    def __call__(self, sample: Dict[str, Any], params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        改进版调用逻辑：
        1. 所有图片必用 base_x_shift=0.15, base_y_shift=0.10
        2. 抽选图片（is_extra=True）额外增加0.1偏移
        """
        if params is None:
            raise ValueError("需传入params，包含：base_x_shift, base_y_shift, shift_range, is_extra")
        
        # 从params获取配置（必传参数）
        base_x = params["base_x_shift"]    # 基础X偏移（固定0.15）
        base_y = params["base_y_shift"]    # 基础Y偏移（固定0.10）
        shift_range = params["shift_range"]# 额外偏移量（固定0.1）
        is_extra = params["is_extra"]      # 是否为抽选图片（True/False）
        
        # 计算总偏移：基础偏移 + （额外偏移 if 抽选 else 0）
        total_x_shift = base_x + (shift_range if is_extra else 0.0)
        total_y_shift = base_y + (shift_range if is_extra else 0.0)
        
        # 获取图像数据 (C,H,W)，转为float
        image = sample["data"].to(dtype=torch.float32)
        
        # 处理多通道（仅调整第一个通道，假设为灰度图）
        if image.shape[0] > 1:
            image_channel = image[0:1].contiguous()   # 保持 (1,H,W) 维度
        else:
            image_channel = image.contiguous() 
        
        # 提取原始图像的统计参数（1%/99%分位数防噪声，25%/75%分位数定位控制点）
        original_min = torch.quantile(image_channel.flatten(), 0.01)  # 原始最小值（1%分位）
        original_max = torch.quantile(image_channel.flatten(), 0.99)  # 原始最大值（99%分位）
        lower_q = torch.quantile(image_channel.flatten(), 0.25)       # 25%分位
        upper_q = torch.quantile(image_channel.flatten(), 0.75)       # 75%分位
        mask_mid = (image_channel >= lower_q) & (image_channel <= upper_q)
        original_control_x = torch.mean(image_channel[mask_mid])      # 原始控制点X坐标
        
        # 计算新控制点（基于总偏移）
        window_width = original_max - original_min  # 图像动态范围
        new_control_x = original_control_x + window_width * total_x_shift  # X方向总偏移
        new_control_x = torch.clamp(new_control_x, original_min + 1e-8, original_max - 1e-8)  # 限制范围
        
        new_control_y = 0.5 + total_y_shift  # Y方向总偏移（原始中点0.5）
        new_control_y = min(max(new_control_y, 0.0 + 1e-8), 1.0 - 1e-8)  # 限制在[0,1]
        
        # 应用控制点调整
        adjusted_image = self.apply_control_point_adjustment(
            image_channel, original_min, original_max, new_control_x, new_control_y
        )
        
        # 回填调整后的通道
        if image.shape[0] > 1:
            image[0:1] = adjusted_image
            image = image.contiguous()
        else:
            image = adjusted_image
        
        sample["data"] = image
        return sample

class RandomScatterArtifact:
    """在2D DRR上模拟X线散射伪影"""
    def __init__(self, intensity_range=[5, 30], count_range=[10, 50]):
        self.intensity_range = intensity_range
        self.count_range = count_range
        
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # 转换为numpy数组进行处理
        drr_img = sample["data"].cpu().numpy().squeeze()
        # 假设输入是归一化的[0,1]，转换为[0,255]进行处理
        drr_img = (drr_img * 255).astype(np.uint8)
        
        h, w = drr_img.shape
        count = np.random.randint(*self.count_range)
        
        for _ in range(count):
            # 随机射线方向（从边缘向中心）
            angle = np.random.uniform(0, np.pi)
            x0 = np.random.choice([0, w-1])
            y0 = np.random.randint(0, h)
            
            # 生成射线沿线的散射条纹
            length = np.random.randint(h//2, h)
            x1 = x0 + int(length * np.cos(angle))
            y1 = y0 + int(length * np.sin(angle))
            
            # 确保坐标在图像范围内
            x1 = np.clip(x1, 0, w-1)
            y1 = np.clip(y1, 0, h-1)
            
            # 在射线周围加模糊亮度
            line = cv2.line(
                np.zeros_like(drr_img), 
                (x0, y0), 
                (x1, y1), 
                color=(1,),  # 单元素元组，符合 cv2.Scalar 类型要求（灰度图）
                thickness=3,
                lineType=cv2.LINE_AA  # 可选：抗锯齿线条
            )
            scatter = gaussian_filter(line, sigma=2) * np.random.uniform(*self.intensity_range)
            drr_img = np.clip(drr_img + scatter, 0, 255)
        
        # 转换回torch张量并归一化到[0,1]
        drr_img = drr_img.astype(np.float32) / 255.0
        sample["data"] = torch.from_numpy(drr_img).unsqueeze(0)  # 恢复通道维度
        return sample


class RandomGlobalContrastStretch:
    """全局对比度拉伸"""
    def __init__(self, contrast_range=[0.5, 2.0]):
        self.contrast_range = contrast_range
        
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # 转换为numpy数组进行处理
        drr_img = sample["data"].cpu().numpy().squeeze()
        factor = np.random.uniform(*self.contrast_range)
        
        # 忽略背景（假设背景为0）
        non_zero_mask = drr_img > 0
        if np.any(non_zero_mask):
            mean = np.mean(drr_img[non_zero_mask])
            # 应用对比度拉伸
            drr_img = (drr_img - mean) * factor + mean
            # 截断并转换回torch张量
            drr_img = np.clip(drr_img, 0.0, 1.0)
            sample["data"] = torch.from_numpy(drr_img).unsqueeze(0)
        
        return sample


class RandomGrayDrift:
    """全局灰度漂移"""
    def __init__(self, drift_range=[-20, 20]):
        self.drift_range = drift_range
        
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # 转换为[0,255]范围处理
        drr_img = (sample["data"].cpu().numpy().squeeze() * 255).astype(np.int32)
        drift = np.random.randint(*self.drift_range)
        
        # 应用灰度漂移
        drr_img = drr_img + drift
        # 截断并转换回[0,1]范围的torch张量
        drr_img = np.clip(drr_img, 0, 255).astype(np.float32) / 255.0
        sample["data"] = torch.from_numpy(drr_img).unsqueeze(0)
        
        return sample