# In cxas/models/UNet/jtr_modules.py (suggested file name)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any # Added missing imports

class JTR_Encoder(nn.Module):
    """
    Encoder for the Joint Task Regularization (JTR) module.
    It processes concatenated segmentation predictions and ground truth/pseudo-labels
    into a compact latent representation.
    """
    def __init__(self, in_channels: int, latent_channels: int = 512) -> None:
        """
        Initializes the JTR Encoder.

        Args:
            in_channels (int): Number of input channels. This will typically be 
                               `num_classes * 2` (for predictions and labels).
            latent_channels (int): The number of channels in the deepest latent representation.
        """
        super().__init__()
        self.in_channels: int = in_channels
        self.latent_channels: int = latent_channels
        '''
        self.attention_branch = nn.Sequential(
            # x的输入是1通道（如[7,1,512,512]），先升到中间维度
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 再升到与y_probs相同的通道数（如54），用Sigmoid确保输出在0~1之间
            nn.Conv2d(32, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()  # 注意力掩码值∈[0,1]（0=不关注，1=重点关注）
        )
            '''
        in_channels +=1
        self.conv_blocks = nn.Sequential(
            # 仅2个卷积块，大倍数下采样（4x）
            nn.Conv2d(in_channels, latent_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels // 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4),  # 一次下采样4x，替代两次2x

            # 直接输出latent_channels，无额外卷积块
            nn.Conv2d(latent_channels // 4, latent_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
        )


    def forward(self,  y_probs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the JTR Encoder.

        Args:
            x (torch.Tensor): Input tensor, typically concatenated predictions and labels
                              (B, C_in, H, W).

        Returns:
            torch.Tensor: The latent representation tensor (B, latent_channels, H', W').
                          The spatial dimensions H', W' depend on the input H, W and the
                          MaxPool2d layers used.
        """

        # 步骤1：用x生成注意力掩码（与y_probs同形状：[B,54,512,512]）
        # attn_mask = self.attention_branch(x)  # x→[B,54,512,512]
        
        # 步骤2：用掩码加权y_probs（x引导编码器关注重要区域）
        #y_attended = y_probs * attn_mask  # 逐元素相乘，实现条件过滤

        concatenated = torch.cat([x, y_probs], dim=1)
        
        # 步骤3：加权后的y送入主干编码器
        latent = self.conv_blocks(concatenated)

        return latent


class JTR_Decoder(nn.Module):
    """
    Decoder for the Joint Task Regularization (JTR) module.
    It reconstructs the input (concatenated predictions and labels)
    from the latent representation produced by JTR_Encoder.
    """
    def __init__(self, out_channels: int, latent_channels: int = 512) -> None:
        """
        Initializes the JTR Decoder.

        Args:
            out_channels (int): Number of output channels for reconstruction.
                                This will typically be `num_classes * 2`.
            latent_channels (int): The number of channels in the input latent representation.
        """
        super().__init__()
        self.out_channels: int = out_channels
        self.latent_channels: int = latent_channels

        self.conv_blocks = nn.Sequential(
            # 输入通道=latent_channels（因融合方式改为加法，无需考虑拼接）
            nn.ConvTranspose2d(latent_channels, latent_channels // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels // 2),
            nn.ReLU(inplace=True),

            # 第二次上采样，输出通道进一步缩减
            nn.ConvTranspose2d(latent_channels // 2, latent_channels // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels // 4),
            nn.ReLU(inplace=True),

            # 最终卷积到目标通道数
            nn.Conv2d(latent_channels // 4, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Performs the forward pass of the JTR Decoder.

        Args:
            x (torch.Tensor): Input latent representation tensor from JTR_Encoder
                                (B, latent_channels, H', W').
            target_size (Tuple[int, int]): The desired output spatial dimensions (H, W).
                                            The decoder will interpolate to this size if needed.

        Returns:
            torch.Tensor: Reconstructed tensor (B, out_channels, target_H, target_W).
        """
        # Apply convolutional transpose blocks
        x = self.conv_blocks(x)

        # Ensure the final output matches the target_size.
        # F.interpolate is used as a final resizing step in case ConvTranspose2d
        # doesn't perfectly match the desired output resolution due to stride/padding.
        # align_corners=False is generally preferred for non-learned interpolation in neural networks.
        if x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            
        return x

class InputFeatureExtractor(nn.Module):
    """
    原始输入 x 的特征提取器，用于将输入图像转换为与 JTR 编码器兼容的特征表示。
    """
    def __init__(self, jtr_latent_channels: int = 512, input_channels: int = 1) -> None:
        """
        初始化输入特征提取器。
        
        Args:
            jtr_latent_channels: JTR 编码器的潜在通道数，用于确定特征维度。
            input_channels: 原始输入 x 的通道数（如图像为 3）。
        """
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            # 仅1个卷积块，快速下采样
            nn.Conv2d(input_channels, jtr_latent_channels, kernel_size=3, padding=1, bias=False),  # 128//8=16通道
            nn.BatchNorm2d(jtr_latent_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4), 
        )
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取输入 x 的特征。
        
        Args:
            x: 原始输入张量，形状为 (B, input_channels, H, W)。
            
        Returns:
            提取的特征张量，形状为 (B, jtr_latent_channels//2, H/4, W/4)。
        """
        return self.feature_extractor(x)