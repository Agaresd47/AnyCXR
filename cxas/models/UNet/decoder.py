# In cxas/models/UNet/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Union # Updated imports for clarity

# Import blocks (make sure ResidualConvBlock and OutConv are correctly defined in unet_blocks.py)
from .unet_blocks import OutConv, ResidualConvBlock
# Import attention modules (make sure these are defined in attention_modules.py)
from .attention_modules import CASModule, AMFSModule

# AngleInjectionModule - copied and aligned with the refined version used in Encoder.
# In a full project, this would ideally be in a shared utility file (e.g., `cxas/modules/angle_injection.py`)
# and imported by both encoder.py and decoder.py to avoid duplication.
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
            # This allows element-wise addition after transformation.
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
            # If angle_embedding_dim was 0 during init, no transform module was created.
            # In this case, simply return the original features.
            return features 
        
        # angle_embedding: (B, angle_embedding_dim)
        # Transformed to (B, feature_channels)
        transformed_angle: torch.Tensor = self.transform(angle_embedding) 
        
        # Spatially broadcast transformed_angle to match feature map dimensions (H, W)
        # From (B, C) to (B, C, 1, 1) for broadcasting
        transformed_angle_for_ops: torch.Tensor = transformed_angle.unsqueeze(2).unsqueeze(3) 

        # Element-wise addition for conditioning:
        # This adds the angle information to each feature map, leveraging broadcasting.
        return features + transformed_angle_for_ops


class Decoder(nn.Module):
    """
    UNet decoder path, including upsampling, skip connections,
    Attention Multi-Feature Selection (AMFS) modules, Cross Attention Selection (CAS) modules,
    and optional angle conditioning.
    """
    def __init__(self, n_classes: int, angle_embedding_dim: int = 0) -> None:
        """
        Initializes the Decoder module.

        Args:
            n_classes (int): The number of output classes for segmentation.
                             You mentioned 55 classes for your model.
            angle_embedding_dim (int): Dimension of the angle embedding. If > 0,
                                       angle conditioning modules are instantiated.
        """
        super().__init__()
        self.n_classes: int = n_classes
        self.angle_embedding_dim: int = angle_embedding_dim 
        
        # Define AMFS dilation rates as per the paper's Decoder 1
        self.amfs_dilation_rates: Tuple[int, int, int] = (1, 2, 6) 


        self.encoder_feature_channels: Dict[str, int] = {
            'feat0': 64,   # Output after initial conv1 block
            'feat1': 256,  # Output of encoder's layer1
            'feat2': 512,  # Output of encoder's layer2
            'feat3': 1024, # Output of encoder's layer3
            'feat4': 2048  # Output of encoder's layer4 (deepest feature)
        }
        
        # --- Decoder processing blocks ---
        # Each decoding stage typically involves:
        # 1. Upsampling the lower-resolution feature map.
        # 2. Combining it with a higher-resolution skip connection (potentially with attention).
        # 3. Passing through a convolutional block (e.g., ResidualConvBlock).
        # 4. Applying AngleInjection (if enabled).
        # 5. Applying Attention Multi-Feature Selection (AMFS) Module.

        # Decoder Stage 4 (Combines feat4 with feat3 skip)
        # Input to decoder_block4 is feat4_channels + feat3_channels
        # Output of decoder_block4 is feat3_channels (downsizing after concatenation)
        self.decoder_block4: ResidualConvBlock = ResidualConvBlock(
            self.encoder_feature_channels['feat4'] + self.encoder_feature_channels['feat3'], 
            self.encoder_feature_channels['feat3']
        )
        self.amfs4: AMFSModule = AMFSModule(
            self.encoder_feature_channels['feat3'], # AMFS input is output of decoder_block4
            self.encoder_feature_channels['feat3'], # AMFS output channels
            self.amfs_dilation_rates
        )

        # Decoder Stage 3 (Combines current x with feat2 skip)
        # Input to decoder_block3 is prev_output_channels + feat2_channels
        self.decoder_block3: ResidualConvBlock = ResidualConvBlock(
            self.encoder_feature_channels['feat3'] + self.encoder_feature_channels['feat2'], 
            self.encoder_feature_channels['feat2']
        )
        self.amfs3: AMFSModule = AMFSModule(
            self.encoder_feature_channels['feat2'], 
            self.encoder_feature_channels['feat2'], 
            self.amfs_dilation_rates
        )

        # Decoder Stage 2 (Combines current x with feat1 skip)
        self.decoder_block2: ResidualConvBlock = ResidualConvBlock(
            self.encoder_feature_channels['feat2'] + self.encoder_feature_channels['feat1'], 
            self.encoder_feature_channels['feat1']
        )
        self.amfs2: AMFSModule = AMFSModule(
            self.encoder_feature_channels['feat1'], 
            self.encoder_feature_channels['feat1'], 
            self.amfs_dilation_rates
        )

        # Decoder Stage 1 (Combines current x with feat0 skip)
        self.decoder_block1: ResidualConvBlock = ResidualConvBlock(
            self.encoder_feature_channels['feat1'] + self.encoder_feature_channels['feat0'], 
            self.encoder_feature_channels['feat0']
        )
        self.amfs1: AMFSModule = AMFSModule(
            self.encoder_feature_channels['feat0'], 
            self.encoder_feature_channels['feat0'], 
            self.amfs_dilation_rates
        )

        # CAS (Cross Attention Selection) Modules:
        # Applied to the skip connections from the encoder.
        # Based on your code, CAS is applied only to the deeper skips (feat3 and feat2).
        self.cas1: CASModule = CASModule(
            encoder_channels=self.encoder_feature_channels['feat3'], # Skip3 channels (from encoder layer3)
            decoder_channels=self.encoder_feature_channels['feat4']  # Channels of the current feature map 'x' (feat4)
        ) 
        self.cas2: CASModule = CASModule(
            encoder_channels=self.encoder_feature_channels['feat2'], # Skip2 channels (from encoder layer2)
            decoder_channels=self.encoder_feature_channels['feat3']  # Channels of the current feature map 'x' (after decoder_block4/amfs4)
        ) 

        # Output Convolution layer to produce final segmentation map
        self.out_conv: OutConv = OutConv(self.encoder_feature_channels['feat0'], self.n_classes)

        # Conditional Angle Injection Modules for Decoder
        self.angle_inject_dec4: Optional[AngleInjectionModule] = None
        self.angle_inject_dec3: Optional[AngleInjectionModule] = None
        self.angle_inject_dec2: Optional[AngleInjectionModule] = None
        self.angle_inject_dec1: Optional[AngleInjectionModule] = None

        if self.angle_embedding_dim > 0:
            # Inject angle embedding into the feature maps after each ResidualConvBlock
            # and before the AMFS module in each decoding stage.
            self.angle_inject_dec4 = AngleInjectionModule(
                feature_channels=self.encoder_feature_channels['feat3'], # Output channels of decoder_block4
                angle_embedding_dim=self.angle_embedding_dim
            )
            self.angle_inject_dec3 = AngleInjectionModule(
                feature_channels=self.encoder_feature_channels['feat2'], # Output channels of decoder_block3
                angle_embedding_dim=self.angle_embedding_dim
            )
            self.angle_inject_dec2 = AngleInjectionModule(
                feature_channels=self.encoder_feature_channels['feat1'], # Output channels of decoder_block2
                angle_embedding_dim=self.angle_embedding_dim
            )
            self.angle_inject_dec1 = AngleInjectionModule(
                feature_channels=self.encoder_feature_channels['feat0'], # Output channels of decoder_block1
                angle_embedding_dim=self.angle_embedding_dim
            )

    def _decode(self, features: Dict[str, torch.Tensor], img_shape: Tuple[int, int], angle_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs the decoding operations of the UNet.

        Args:
            features (Dict[str, torch.Tensor]): A dictionary of feature maps from the encoder
                                                with keys 'feat0', 'feat1', 'feat2', 'feat3', 'feat4'.
            img_shape (Tuple[int, int]): The original (height, width) of the input image.
                                         Used for final interpolation to match resolution.
            angle_embedding (Optional[torch.Tensor]): Angle embedding tensor (B, angle_embedding_dim).
                                                       Required if angle_embedding_dim > 0.

        Returns:
            torch.Tensor: The final segmentation output tensor (B, n_classes, H, W).

        Raises:
            ValueError: If angle encoding is enabled but 'angle_embedding' is not provided.
        """
        x: torch.Tensor = features['feat4'] # Deepest features from encoder
        skip3: torch.Tensor = features['feat3'] # Skip connection from encoder Layer 3
        skip2: torch.Tensor = features['feat2'] # Skip connection from encoder Layer 2
        skip1: torch.Tensor = features['feat1'] # Skip connection from encoder Layer 1
        skip0: torch.Tensor = features['feat0'] # Skip connection from encoder Conv1 output

        # Ensure angle_embedding is provided if angle conditioning is enabled
        if self.angle_embedding_dim > 0 and angle_embedding is None:
            raise ValueError("Angle encoding enabled, but 'angle_embedding' was not provided to Decoder._decode().")

        # --- Decoder Path ---

        # Stage 4: Upsample feat4, combine with feat3 (after CAS), process, inject angle, AMFS
        x_upsampled_1: torch.Tensor = F.interpolate(x, size=skip3.shape[2:], mode='bilinear', align_corners=True)
        
        # Apply CAS to skip3 using x_upsampled_1 as query/key
        attended_skip3: torch.Tensor = self.cas1(skip3, x_upsampled_1) 
        x_combined_1: torch.Tensor = torch.cat([attended_skip3, x_upsampled_1], dim=1) 
        
        x = self.decoder_block4(x_combined_1)
        if self.angle_inject_dec4: # Inject angle after block
            x = self.angle_inject_dec4(x, angle_embedding) # type: ignore [arg-type]
        x = self.amfs4(x)

        # Stage 3: Upsample current x, combine with feat2 (after CAS), process, inject angle, AMFS
        x_upsampled_2: torch.Tensor = F.interpolate(x, size=skip2.shape[2:], mode='bilinear', align_corners=True)
        
        # Apply CAS to skip2 using x_upsampled_2 as query/key
        attended_skip2: torch.Tensor = self.cas2(skip2, x_upsampled_2)
        x_combined_2: torch.Tensor = torch.cat([attended_skip2, x_upsampled_2], dim=1)
        
        x = self.decoder_block3(x_combined_2)
        if self.angle_inject_dec3: # Inject angle after block
            x = self.angle_inject_dec3(x, angle_embedding) # type: ignore [arg-type]
        x = self.amfs3(x)

        # Stage 2: Upsample current x, combine with feat1 (no CAS), process, inject angle, AMFS
        x_upsampled_3: torch.Tensor = F.interpolate(x, size=skip1.shape[2:], mode='bilinear', align_corners=True)
        x_combined_3: torch.Tensor = torch.cat([skip1, x_upsampled_3], dim=1) # Standard concatenation
        
        x = self.decoder_block2(x_combined_3)
        if self.angle_inject_dec2: # Inject angle after block
            x = self.angle_inject_dec2(x, angle_embedding) # type: ignore [arg-type]
        x = self.amfs2(x)

        # Stage 1: Upsample current x, combine with feat0 (no CAS), process, inject angle, AMFS
        x_upsampled_4: torch.Tensor = F.interpolate(x, size=skip0.shape[2:], mode='bilinear', align_corners=True)
        x_combined_4: torch.Tensor = torch.cat([skip0, x_upsampled_4], dim=1) # Standard concatenation
        
        x = self.decoder_block1(x_combined_4)
        if self.angle_inject_dec1: # Inject angle after block
            x = self.angle_inject_dec1(x, angle_embedding) # type: ignore [arg-type]
        x = self.amfs1(x)
        
        # Ensure the final feature map 'x' has the exact original image dimensions (img_shape)
        # This handles cases where scaling might not perfectly match due to downsampling/upsampling
        if x.shape[2:] != img_shape:
            x = F.interpolate(x, size=img_shape, mode='bilinear', align_corners=True)
        
        # Final output convolution to get per-class logits
        return self.out_conv(x)

    def forward(self, features: Dict[str, torch.Tensor], img_shape: Tuple[int, int], angle_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Defines the computation performed at every call for the Decoder.
        This method primarily delegates to the internal _decode method.

        Args:
            features (Dict[str, torch.Tensor]): A dictionary of feature maps from the encoder.
            img_shape (Tuple[int, int]): The original (height, width) of the input image.
            angle_embedding (Optional[torch.Tensor]): Angle embedding tensor.

        Returns:
            torch.Tensor: The final segmentation output tensor.
        """
        return self._decode(features, img_shape, angle_embedding)