# In cxas/models/UNet/encoder.py

import torch.nn as nn
import torch
from typing import Optional, List, Tuple, Dict, Union # Updated imports for clarity
from collections import OrderedDict

# Import the functions from the new backbone_adapters.py
from .backbone_adapters import get_resnet_backbone, get_vgg_backbone 

# Helper module for angle injection (can be an inner class or defined globally in this file)
class AngleInjectionModule(nn.Module):
    """
    Module for injecting angle embeddings into feature maps.
    It transforms the angle embedding and adds it element-wise to the features,
    broadcasting across spatial dimensions.
    """
    def __init__(self, feature_channels: int, angle_embedding_dim: int) -> None:
        """
        Initializes the AngleInjectionModule.

        Args:
            feature_channels (int): The number of channels in the feature maps to be conditioned.
            angle_embedding_dim (int): The dimension of the input angle embedding.
                                       If 0, no transformation or injection occurs.
        """
        super().__init__()
        self.feature_channels: int = feature_channels
        self.angle_embedding_dim: int = angle_embedding_dim

        self.transform: Optional[nn.Sequential] = None
        if angle_embedding_dim > 0:
            # A small MLP to transform the angle embedding to match feature_channels
            # This allows element-wise addition without size mismatch
            self.transform = nn.Sequential(
                nn.Linear(angle_embedding_dim, feature_channels),
                nn.ReLU(inplace=True),
            )

            self.transform = nn.Sequential(
                nn.Linear(angle_embedding_dim, feature_channels),
                nn.ReLU(inplace=True),
            )


    def forward(self, features: torch.Tensor, angle_embedding: torch.Tensor) -> torch.Tensor:
        """
        Applies angle injection to the feature maps.

        Args:
            features (torch.Tensor): Input feature maps (B, C, H, W).
            angle_embedding (torch.Tensor): Angle embedding (B, angle_embedding_dim).

        Returns:
            torch.Tensor: Conditioned feature maps (B, C, H, W).
        """
        if self.transform is None:
            return features # No angle injection if angle_embedding_dim was 0

        # angle_embedding: (B, angle_embedding_dim)
        # Transformed to (B, feature_channels)
        transformed_angle: torch.Tensor = self.transform(angle_embedding) 
        
        # Spatially broadcast transformed_angle to match feature map dimensions (H, W)
        # From (B, C) to (B, C, 1, 1) for broadcasting
        transformed_angle_for_ops: torch.Tensor = transformed_angle.unsqueeze(2).unsqueeze(3) 

        # Element-wise addition for conditioning:
        # This adds the angle information to each feature map.
        # Broadcasting handles the spatial dimensions (H, W).
        return features + transformed_angle_for_ops


class Encoder(nn.Module):
    """
    UNet encoder with pretrained backbones (e.g., ResNet) adapted for 1-channel input
    and optional angle conditioning.
    """

    def _build_encoder(self) -> nn.Module:
        """
        Constructs the backbone encoder with 1-channel input adaptation.

        Returns:
            nn.Module: The sequential encoder module.
        Raises:
            NotImplementedError: If the specified backbone network is not supported.
        """
        if "resnet" in self.network_name:
            # Call the function from backbone_adapters.py
            return get_resnet_backbone(self.network_name)
        elif "vgg" in self.network_name:
            # As per user's instruction, VGG is not cared about for now.
            raise NotImplementedError(f"VGG backbone ('{self.network_name}') is not implemented yet.")
        else:
            raise NotImplementedError(f"Backbone '{self.network_name}' not supported.")

    def __init__(self, network_name: str, angle_embedding_dim: int = 0) -> None:
        """
        Initializes the Encoder module.

        Args:
            network_name (str): Name of the backbone network (e.g., "resnet18", "resnet50").
            angle_embedding_dim (int): Dimension of the angle embedding. If > 0,
                                       angle conditioning modules are instantiated.
        """
        super().__init__()
        self.network_name: str = network_name.lower()
        self.encoder: nn.Module = self._build_encoder()
        self.angle_embedding_dim: int = angle_embedding_dim

        # Angle injection modules are Optional as they depend on angle_embedding_dim
        self.angle_inject_conv1: Optional[AngleInjectionModule] = None
        self.angle_inject_layer1: Optional[AngleInjectionModule] = None
        self.angle_inject_layer2: Optional[AngleInjectionModule] = None
        self.angle_inject_layer3: Optional[AngleInjectionModule] = None
        self.angle_inject_layer4: Optional[AngleInjectionModule] = None

        if self.angle_embedding_dim > 0:
            # Determine expansion factor for ResNet bottleneck blocks
            expansion_factor: int = self._get_expansion_factor()
            
            # Instantiate AngleInjectionModules for each feature level
            # These channel numbers are standard for ResNet series.
            # conv1 output channels typically 64.
            self.angle_inject_conv1 = AngleInjectionModule(feature_channels=64, angle_embedding_dim=self.angle_embedding_dim)
            # layer1 output channels typically 64 * expansion_factor
            self.angle_inject_layer1 = AngleInjectionModule(feature_channels=64 * expansion_factor, angle_embedding_dim=self.angle_embedding_dim)
            # layer2 output channels typically 128 * expansion_factor
            self.angle_inject_layer2 = AngleInjectionModule(feature_channels=128 * expansion_factor, angle_embedding_dim=self.angle_embedding_dim)
            # layer3 output channels typically 256 * expansion_factor
            self.angle_inject_layer3 = AngleInjectionModule(feature_channels=256 * expansion_factor, angle_embedding_dim=self.angle_embedding_dim)
            # layer4 output channels typically 512 * expansion_factor
            self.angle_inject_layer4 = AngleInjectionModule(feature_channels=512 * expansion_factor, angle_embedding_dim=self.angle_embedding_dim)

    def _get_expansion_factor(self) -> int:
        """
        Helper to get the expansion factor for ResNet bottleneck blocks.
        (1 for ResNet18/34, 4 for ResNet50/101/152).
        This assumes standard torchvision ResNet models.

        Returns:
            int: The expansion factor of the ResNet blocks.
        """
        if "resnet18" in self.network_name or "resnet34" in self.network_name:
            return 1
        elif "resnet50" in self.network_name or "resnet101" in self.network_name or "resnet152" in self.network_name:
            return 4
        return 1 # Default for other non-bottleneck models or VGG (though VGG is NotImplemented)

    def forward(self, x: torch.Tensor, angle_embedding: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the encoder, returning features at multiple scales,
        with optional angle conditioning.

        Args:
            x (torch.Tensor): Input image tensor (B, 1, H, W) for 1-channel input.
            angle_embedding (Optional[torch.Tensor]): Angle embedding tensor (B, angle_embedding_dim).
                                                       Required if angle_embedding_dim > 0.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing feature maps at different scales.
                                     Keys are 'feat0', 'feat1', 'feat2', 'feat3', 'feat4'.

        Raises:
            ValueError: If angle encoding is enabled but 'angle_embedding' is not provided.
        """
        features: Dict[str, torch.Tensor] = {}

        # Ensure angle_embedding is provided if angle conditioning is enabled
        if self.angle_embedding_dim > 0 and angle_embedding is None:
            raise ValueError("Angle encoding enabled, but 'angle_embedding' was not provided to Encoder.forward().")
        
        # Access layers using the sequential structure of self.encoder
        # self.encoder is an OrderedDict-based Sequential module from get_resnet_backbone

        # Level 0 (Initial convolution, batch norm, relu)
        # Assuming conv1, bn1, relu are grouped under 'conv1' key in the Sequential
        x = self.encoder.conv1(x) 
        if self.angle_inject_conv1: # If angle injection is enabled
            x = self.angle_inject_conv1(x, angle_embedding) # type: ignore [arg-type]
        features['feat0'] = x

        # Maxpool after conv1, before layer1
        x = self.encoder.maxpool(x) 
        
        # Layer 1
        x = self.encoder.layer1(x)
        if self.angle_inject_layer1:
            x = self.angle_inject_layer1(x, angle_embedding) # type: ignore [arg-type]
        features['feat1'] = x

        # Layer 2
        x = self.encoder.layer2(x)
        if self.angle_inject_layer2:
            x = self.angle_inject_layer2(x, angle_embedding) # type: ignore [arg-type]
        features['feat2'] = x

        # Layer 3
        x = self.encoder.layer3(x)
        if self.angle_inject_layer3:
            x = self.angle_inject_layer3(x, angle_embedding) # type: ignore [arg-type]
        features['feat3'] = x

        # Layer 4 (Deepest features for the decoder)
        x = self.encoder.layer4(x)
        if self.angle_inject_layer4:
            x = self.angle_inject_layer4(x, angle_embedding) # type: ignore [arg-type]
        features['feat4'] = x 

        return features



# -------------------------- 密集连接核心组件 --------------------------
class DenseLayer(nn.Module):
    """密集连接块中的基础层：BN + ReLU + 1x1卷积（瓶颈层） + 3x3卷积"""
    def __init__(self, in_channels: int, growth_rate: int, bn_size: int = 4):
        super().__init__()
        # 瓶颈层：1x1卷积减少通道数，降低计算量（论文中隐含此设计）
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1, padding=0, bias=False)
        
        # 3x3卷积提取特征（输入为瓶颈层输出）
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 密集连接：输入包含所有前置层的特征拼接
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        # 与输入特征拼接（密集连接核心）
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    """密集连接块：由多个DenseLayer组成，特征在层间密集传递"""
    def __init__(self, num_layers: int, in_channels: int, growth_rate: int, bn_size: int = 4):
        super().__init__()
        layers = []
        for i in range(num_layers):
            # 每层输入通道数 = 初始通道 + 前i层的增长率累加（密集连接特性）
            layer = DenseLayer(
                in_channels + i * growth_rate,
                growth_rate,
                bn_size
            )
            layers.append(layer)
        self.layers = nn.Sequential(*layers)
        # 记录输出通道数（供后续层参考）
        self.out_channels = in_channels + num_layers * growth_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransitionLayer(nn.Module):
    """过渡层：在DenseBlock之间做下采样+通道压缩，对应论文中“层间max pooling”"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 下采样

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(self.relu(self.bn(x)))
        return self.pool(x)

'''
# -------------------------- 自定义DenseNet编码器 --------------------------
class CustomDenseNetEncoder(nn.Module):
    """
    基于论文描述的自定义DenseNet编码器：
    - 5层结构，第一层为2个3x3卷积，第2-5层为密集连接块（6/12/24/16层）
    - 支持单通道输入（CXR图像）、角度嵌入
    - 输出与原有Encoder相同格式的特征字典（'feat0'-'feat4'）
    """
    def __init__(self, angle_embedding_dim: int = 0, growth_rate: int = 24):
        super().__init__()
        self.angle_embedding_dim = angle_embedding_dim
        self.growth_rate = growth_rate  # 密集连接的增长率（控制通道增长速度）

        # -------------------------- 1. 第一层：2个3x3卷积（非密集连接） --------------------------
        # 输入为单通道（CXR），初始通道数设为64（可调整）
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),  # 第二个3x3卷积
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool_after_conv1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 第一层后的下采样

        # -------------------------- 2. 第2-5层：密集连接块（DenseBlock） --------------------------
        # 第2层：6个密集连接层
        self.dense_block1 = DenseBlock(
            in_channels=64,  # 接收第一层输出的64通道
            num_layers=6,
            growth_rate=growth_rate
        )
        # 过渡层1：压缩通道并下采样（输出通道设为密集块输出的一半，符合论文“CAS模块维度调整”逻辑）
        self.transition1 = TransitionLayer(
            in_channels=self.dense_block1.out_channels,
            out_channels=self.dense_block1.out_channels // 4
        )

        # 第3层：12个密集连接层
        self.dense_block2 = DenseBlock(
            in_channels=self.transition1.conv.out_channels,  # 接收过渡层1的输出
            num_layers=12,
            growth_rate=growth_rate
        )
        self.transition2 = TransitionLayer(
            in_channels=self.dense_block2.out_channels,
            out_channels=self.dense_block2.out_channels // 4
        )

        # 第4层：24个密集连接层
        self.dense_block3 = DenseBlock(
            in_channels=self.transition2.conv.out_channels,
            num_layers=16,
            growth_rate=growth_rate
        )
        self.transition3 = TransitionLayer(
            in_channels=self.dense_block3.out_channels,
            out_channels=self.dense_block3.out_channels // 4
        )

        # 第5层：16个密集连接层（最后一层无过渡层，直接输出）
        self.dense_block4 = DenseBlock(
            in_channels=self.transition3.conv.out_channels,
            num_layers=12,
            growth_rate=growth_rate
        )
        self.feature_channels = self._compute_feature_channels() 
        # -------------------------- 3. 角度嵌入模块 --------------------------
        # 与原有Encoder保持一致，在各特征层后注入角度信息
        self.angle_inject_feat0 = AngleInjectionModule(64, angle_embedding_dim) if angle_embedding_dim > 0 else None
        self.angle_inject_feat1 = AngleInjectionModule(self.transition1.conv.out_channels, angle_embedding_dim) if angle_embedding_dim > 0 else None
        self.angle_inject_feat2 = AngleInjectionModule(self.transition2.conv.out_channels, angle_embedding_dim) if angle_embedding_dim > 0 else None
        self.angle_inject_feat3 = AngleInjectionModule(self.transition3.conv.out_channels, angle_embedding_dim) if angle_embedding_dim > 0 else None
        self.angle_inject_feat4 = AngleInjectionModule(self.dense_block4.out_channels, angle_embedding_dim) if angle_embedding_dim > 0 else None

    def _compute_feature_channels(self) -> Dict[str, int]:
        """计算并返回各特征层的通道数（与forward输出的feat对应）"""
        # feat0：conv1输出（64通道）
        feat0_channels = 64

        # feat1：transition1输出（密集块1输出 // 4）
        dense1_out = 64 + 6 * self.growth_rate  # dense_block1的输出通道数
        feat1_channels = dense1_out // 4  # transition1的out_channels

        # feat2：transition2输出（密集块2输出 // 4）
        dense2_out = feat1_channels + 12 * self.growth_rate  # dense_block2的输入是feat1_channels
        feat2_channels = dense2_out // 4  # transition2的out_channels

        # feat3：transition3输出（密集块3输出 // 4）
        dense3_out = feat2_channels + 16 * self.growth_rate  # dense_block3的输入是feat2_channels
        feat3_channels = dense3_out // 4  # transition3的out_channels

        # feat4：dense_block4输出
        feat4_channels = feat3_channels + 12 * self.growth_rate  # dense_block4的输入是feat3_channels

        return {
            'feat0': feat0_channels,
            'feat1': feat1_channels,
            'feat2': feat2_channels,
            'feat3': feat3_channels,
            'feat4': feat4_channels
        }

    def forward(self, x: torch.Tensor, angle_embedding: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播：返回各层特征字典，与原有Encoder兼容
        Args:
            x: 输入单通道图像 (B, 1, H, W)
            angle_embedding: 角度嵌入 (B, angle_embedding_dim)，角度嵌入开启时必传
        Returns:
            特征字典：'feat0'（第一层输出）、'feat1'-'feat4'（第2-5层输出）
        """
        features = {}


        # 检查角度嵌入是否合法
        if self.angle_embedding_dim > 0 and angle_embedding is None:
            raise ValueError("Angle encoding enabled, but 'angle_embedding' not provided.")

        # -------------------------- 第一层特征：feat0 --------------------------
        x = self.conv1(x)  # (B, 64, H, W)
        if self.angle_inject_feat0:
            x = self.angle_inject_feat0(x, angle_embedding)  # 注入角度信息
        features['feat0'] = x  # 保存第一层特征



        # 第一层后的下采样（进入第2层）
        x = self.maxpool_after_conv1(x)

        # -------------------------- 第2层特征：feat1 --------------------------
        x = self.dense_block1(x)  # 密集连接块1输出
        x = self.transition1(x)   # 过渡层下采样
        if self.angle_inject_feat1:
            x = self.angle_inject_feat1(x, angle_embedding)
        features['feat1'] = x

        # -------------------------- 第3层特征：feat2 --------------------------
        x = self.dense_block2(x)
        x = self.transition2(x)
        if self.angle_inject_feat2:
            x = self.angle_inject_feat2(x, angle_embedding)
        features['feat2'] = x


        # -------------------------- 第4层特征：feat3 --------------------------
        x = self.dense_block3(x)
        x = self.transition3(x)
        if self.angle_inject_feat3:
            x = self.angle_inject_feat3(x, angle_embedding)
        features['feat3'] = x


        # -------------------------- 第5层特征：feat4 --------------------------
        x = self.dense_block4(x)  # 最后一层无过渡层
        if self.angle_inject_feat4:
            x = self.angle_inject_feat4(x, angle_embedding)
        features['feat4'] = x


        return features

'''