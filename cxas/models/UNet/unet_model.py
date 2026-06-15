# In cxas/models/UNet/unet_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any, Tuple, Dict, Union # Added Union for forward return type

from .encoder import Encoder
from .decoder import Decoder
from .jtr_modules import JTR_Encoder, JTR_Decoder, InputFeatureExtractor # Updated import path for JTR modules
from cxas.losses.compound_losses import DC_and_BCE_loss # Assuming this is your compound loss function


class UNet(nn.Module):
    """
    A Conditional UNet model with optional Joint Task Regularization (JTR).
    This model is designed to intake all views of a single CXR and return 55 classes of output.
    """
    def __init__(self,
                    model_name: str,
                    in_channels: int,
                    n_classes: int, # Expected to be 55 as per your requirement
                    angle_encoding_type: str = "sin_cos",
                    enable_jtr: bool = False, 
                    jtr_latent_channels: int = 128
                ) -> None:
        """
        Initializes the UNet model.

        Args:
            model_name (str): Name of the backbone encoder model (e.g., "resnet50").
            in_channels (int): Number of input channels for the image (e.g., 1 for grayscale CXR).
            n_classes (int): Number of output classes for segmentation (e.g., 55).
            angle_encoding_type (str): Type of angle encoding ("sin_cos", "raw_normalized", or other).
                                        "sin_cos" results in angle_embedding_dim=2.
                                        "raw_normalized" results in angle_embedding_dim=1.
                                        Anything else or if view_angle is None will result in angle_embedding_dim=0.
            enable_jtr (bool): If True, enables the Joint Task Regularization module.
            jtr_latent_channels (int): Number of channels in the latent space of JTR encoder/decoder.
        """
        super().__init__()
        
        self.angle_encoding_type: str = angle_encoding_type
        # Determine angle_embedding_dim based on the encoding type
        self.angle_embedding_dim: int = 0
        if angle_encoding_type == "sin_cos":
            self.angle_embedding_dim = 2
        elif angle_encoding_type == "raw_normalized":
            self.angle_embedding_dim = 1

        # Encoder: Takes image data and conditional features (angle, etc.)
        self.encoder: Encoder = Encoder(model_name, angle_embedding_dim=self.angle_embedding_dim)
        #self.encoder: CustomDenseNetEncoder = CustomDenseNetEncoder(angle_embedding_dim=self.angle_embedding_dim)
        
        # Decoder: Produces n_classes output channels
        self.decoder: Decoder = Decoder( n_classes=n_classes, angle_embedding_dim=self.angle_embedding_dim)

        # Segmentation loss function setup (DC_and_BCE_loss)
        bce_kwargs: Dict[str, Any] = {'reduction': 'none'} # Force reduction to 'none' for manual weighting
        soft_dice_kwargs: Dict[str, Any] = {
            'do_bg': True,       # Include background in Dice calculation.
                                    # If a background class exists and its weight is 0.0, it's effectively ignored.
                                    # This is safer than `do_bg: False` if class 0 isn't strictly background.
            'batch_dice': False, # Critical: Set to False for per-sample class weighting with partial labels
            'smooth': 1e-5,      # Small epsilon for numerical stability.
            'ddp': False         # Set to True if training with Distributed Data Parallel.
                                    # The trainer should handle DDP setup and pass this correctly.
        }
        self.seg_loss_fn: DC_and_BCE_loss = DC_and_BCE_loss(
                bce_kwargs=bce_kwargs,
                soft_dice_kwargs=soft_dice_kwargs,
                weight_ce=1,      # Weight for BCE component of segmentation loss
                weight_dice=1,    # Weight for Dice component of segmentation loss
                use_ignore_label=False, # Set to True if your ground truth `target` tensor includes an explicit ignore channel.
        )

        self.n_classes: int = n_classes
        self.in_channels: int = in_channels

        self.enable_jtr: bool = enable_jtr 
        self.jtr_latent_channels: int = jtr_latent_channels

        # Initialize JTR encoder and decoder conditionally
        self.jtr_encoder: Optional[JTR_Encoder] = None
        self.jtr_decoder: Optional[JTR_Decoder] = None
        if self.enable_jtr:
            # （1）JTR编码器：简化结构，输入通道为n_classes（分割概率图的通道数）
            self.jtr_encoder = JTR_Encoder(
                in_channels=n_classes,  # 输入是分割概率图（B, n_classes, H, W）
                latent_channels=self.jtr_latent_channels  # 128
            )

            # （2）JTR解码器：适配简化结构，输出通道为n_classes（重建概率图）
            self.jtr_decoder = JTR_Decoder(
                out_channels=n_classes,  # 输出与输入概率图通道一致
                latent_channels=self.jtr_latent_channels  # 128，且解码器输入通道已改为与latent_channels一致（因融合方式改为加法）
            )


    def _construct_jtr_target_Y_x(self,
                                    hat_Y_x_probs: torch.Tensor,  # 模型预测的概率分布 (N, C, H, W)
                                    label_indices: torch.Tensor,  # 真实标签的二进制掩码 (N, C, H, W)
                                    class_weights: torch.Tensor  # 类别权重 (N, C)
                                    ) -> torch.Tensor:
        """
        构建JTR的Y_x目标张量，结合类别权重选择性地使用GT或模型预测。
        """
        # 初始化Y_x为模型预测的概率分布
        Y_x_probs: torch.Tensor = hat_Y_x_probs.detach().clone()
        
        # 将标签转换为浮点数格式
        label_one_hot_probs: torch.Tensor = label_indices.float()
        
        # 创建一个基础掩码，默认为所有位置都使用GT
        use_gt_mask: torch.Tensor = torch.ones_like(label_one_hot_probs, dtype=torch.bool)
        
        # 确保类别权重维度正确 (N, C) -> (N, C, 1, 1)
        class_weights_expanded = class_weights.unsqueeze(-1).unsqueeze(-1)  # (N, C, 1, 1)
        
        # 创建权重掩码：权重>0的类别才使用GT
        weight_mask = (class_weights_expanded > 0).expand_as(label_one_hot_probs)
        
        # 更新掩码：只有当权重>0时才使用GT
        use_gt_mask &= weight_mask
        
        # 根据最终掩码选择性地使用GT
        Y_x_probs[use_gt_mask] = label_one_hot_probs[use_gt_mask]
        
        return Y_x_probs

    def forward(self, x: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Performs the forward pass of the UNet model.

        Args:
            x (Dict[str, Any]): A dictionary containing input tensors and metadata:
                - 'data' (torch.Tensor): Input image tensor (N, C_in, H, W).
                - 'label' (Optional[torch.Tensor]): Ground truth segmentation label.
                                                        Shape (N, H, W) (class indices) or
                                                        (N, C_total, H, W) (one-hot, float).
                                                        Used for supervised loss and JTR.
                - 'class_weights' (Optional[torch.Tensor]): Per-class weights for loss calculation (C_total,).
                - 'view_angle' (Optional[torch.Tensor]): View angle for conditional encoding (N,) or (N, 1).
                - 'labeled_task_masks' (Optional[torch.Tensor]): Boolean mask for JTR,
                                                                    indicating labeled regions/classes (N, C, H, W) or (N, C).
                - 'jtr_c_dist_weight' (float): Weight for JTR distance loss component. Default 0.0.
                - 'jtr_c_recon_weight' (float): Weight for JTR reconstruction loss component. Default 0.0.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing various outputs:
                - 'logits' (torch.Tensor): Final segmentation logits (N, n_classes, H, W).
                - 'probabilities' (torch.Tensor): Softmax probabilities of the segmentation (N, n_classes, H, W).
                - 'seg_loss' (Optional[torch.Tensor]): Supervised segmentation loss (scalar).
                - 'jtr_dist_loss' (Optional[torch.Tensor]): JTR distance loss (scalar).
                - 'jtr_recon_loss' (Optional[torch.Tensor]): JTR reconstruction loss (scalar).
                - 'loss' (torch.Tensor): Total combined loss (scalar).
        """
        # Unpack inputs from the dictionary with type hints and default values
        img: torch.Tensor = x['data']
        label: Optional[torch.Tensor] = x.get('label', None)  # Optional ground truth label tensor
        class_weights: Optional[torch.Tensor] = x.get('class_weights', None)
        view_angle: torch.Tensor = x['view_angle']
        
        jtr_c_dist_weight: float = x.get('jtr_c_dist_weight', 0.0) 
        jtr_c_recon_weight: float = x.get('jtr_c_recon_weight', 0.0) 

        original_spatial_shape: Tuple[int, int] = img.shape[2:] 

        # Initialize outputs dictionary
        output: Dict[str, torch.Tensor] = {}
        total_loss: torch.Tensor = torch.tensor(0.0, device=img.device) # Ensure loss is on correct device

        # 1. Angle Encoding (for Conditional UNet)
        angle_embedding: Optional[torch.Tensor] = None
        if self.angle_embedding_dim > 0 and view_angle is not None:
            if self.angle_encoding_type == "sin_cos":
                # Ensure angle is float and convert to radians before sin/cos
                angle_rad: torch.Tensor = torch.deg2rad(view_angle.float()) 
                sin_angle: torch.Tensor = torch.sin(angle_rad)
                cos_angle: torch.Tensor = torch.cos(angle_rad)
                angle_embedding = torch.stack([sin_angle, cos_angle], dim=1) # (N, 2)
            elif self.angle_encoding_type == "raw_normalized":
                angle_embedding = view_angle.float().unsqueeze(1) # (N, 1)
            # Add other encoding types if needed here
            else:
                raise ValueError(f"Unsupported angle_encoding_type: {self.angle_encoding_type}")

        # 2. Main Segmentation Forward Pass
        # Encoder and Decoder should handle `angle_embedding` internally
        features: Dict[str, torch.Tensor] = self.encoder(img, angle_embedding)
        logits: torch.Tensor = self.decoder(features, original_spatial_shape, angle_embedding) # (N, n_classes, H, W)
        output['logits'] = logits

        # This is the line responsible for producing output in the [0, 1] range:
        probabilities: torch.Tensor = F.softmax(logits, dim=1) # Convert logits to probabilities
        output['probabilities'] = probabilities

        # 3. Supervised Segmentation Loss (L_SL)
        # seg_loss_fn handles weighted CE+Dice, expects logits and target.
        if label is not None: # Only calculate if ground truth labels are provided for this batch
            seg_total_loss: torch.Tensor 
            ce_loss_scaled: torch.Tensor
            dc_loss: torch.Tensor
            seg_total_loss, ce_loss_scaled, dc_loss = self.seg_loss_fn(logits, label, class_weights)
            output['seg_loss'] = seg_total_loss
            output['ce_loss'] = ce_loss_scaled
            output['dc_loss'] = dc_loss
            total_loss += seg_total_loss

        # 4. Joint-Task Regularization (JTR)
        # JTR is active IF:
        #   - It was enabled at UNet construction (self.enable_jtr is True)
        #   - JTR encoder/decoder modules exist (checked by `is not None`)
        #   - A ground truth `label` is provided (as JTR needs 'Y_x' which uses GT where available)
        #   - At least one of its dynamic weights is non-zero (not in a warm-up phase or entirely disabled)
        if self.enable_jtr and self.jtr_encoder is not None and self.jtr_decoder is not None \
            and label is not None and (jtr_c_dist_weight > 0.0 or jtr_c_recon_weight > 0.0):
            
            # hat_Y_x (predictions tensor): probabilities of model's current output
            # Now directly using the 'probabilities' already computed and put in output
            hat_Y_x_probs: torch.Tensor = probabilities 

            # Y_x (reliable target tensor): mix of GT (where labeled) and detached predictions (where unlabeled)
            Y_x_probs: torch.Tensor = self._construct_jtr_target_Y_x(
                hat_Y_x_probs=hat_Y_x_probs,
                label_indices=label, # Pass the original label (N, H, W)
                class_weights=class_weights
            )
            
            # --- L_Dist: Distance between latent embeddings of predictions and reliable targets ---
            # Encoder inputs probabilities to JTR encoder
            jtr_latent_hat: torch.Tensor = self.jtr_encoder(hat_Y_x_probs, img) 
            jtr_latent_Y: torch.Tensor = self.jtr_encoder(Y_x_probs, img) 
            
            jtr_latent_hat_flat = jtr_latent_hat.reshape(jtr_latent_hat.size(0), -1)
            jtr_latent_Y_flat = jtr_latent_Y.reshape(jtr_latent_Y.size(0), -1)
            L_dist = (1 - F.cosine_similarity(jtr_latent_hat_flat, jtr_latent_Y_flat, dim=1, eps=1e-8)).mean() / 2.0
            output['L_dist'] = L_dist
            total_loss += jtr_c_dist_weight * L_dist 


            # --- L_Recon: Reconstruction loss for JTR autoencoder ---
            # Targets for reconstruction are detached copies of the inputs to prevent trivial solutions.
            # JTR decoder reconstructs to original_spatial_shape
            recon_from_hat_Y_x = self.jtr_decoder(jtr_latent_hat, original_spatial_shape)
            recon_from_Y_x = self.jtr_decoder(jtr_latent_Y, original_spatial_shape)

            # Use MSE loss for reconstruction, as outputs are probability maps (0-1 range)
            L_recon_hat: torch.Tensor = F.mse_loss(recon_from_hat_Y_x, hat_Y_x_probs.detach())
            L_recon_Y: torch.Tensor = F.mse_loss(recon_from_Y_x, Y_x_probs.detach())
            L_recon: torch.Tensor = L_recon_hat + L_recon_Y

            total_loss += jtr_c_recon_weight * L_recon
            output['L_recon'] = L_recon

        output['loss'] = total_loss # Assign the final total loss
        return output