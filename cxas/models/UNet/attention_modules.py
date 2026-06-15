# In cxas/models/UNet/attention_modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AMFSModule(nn.Module):
    """
    Attention-guided Multi-scale Feature Selection (AMFS) Module.
    Uses dilated convolutions of various sizes to capture multi-scale features,
    enhancing segmentation performance. Each branch has an SEB.
    """
    def __init__(self, in_channels, out_channels, dilation_rates, reduction=16):
        super().__init__()
        assert len(dilation_rates) == 3, "AMFS expects exactly 3 dilation rates."
        
        self.branches = nn.ModuleList()
        for dilation in dilation_rates:
            branch = nn.Sequential(
                SEBlock(in_channels, reduction=reduction), # SEB module on input to branch
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation), # Dilated convolution
                nn.BatchNorm2d(out_channels), # Common to add BN after conv
                nn.ReLU(inplace=True) # Common to add ReLU after BN
            )
            self.branches.append(branch)
            
        # Final 1x1 conv to combine features from all branches
        # The paper says "a 1x1 Convolutional layer with weighted standardization" for reduction,
        # but the AMFS diagram (Fig. 3) usually implies a simple combination after branches.
        # I'll use a standard 1x1 conv here. You can add weight_norm if needed.
        self.combine_conv = nn.Conv2d(out_channels * len(dilation_rates), out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True) # Activation after combination

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        concatenated_features = torch.cat(branch_outputs, dim=1) # Concatenate features from all branches
        return self.relu(self.combine_conv(concatenated_features)) # Fuse branches and apply activation




class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block."""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # Global average pooling
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False), # First fully connected layer (reduction)
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), # Second fully connected layer (expansion)
            nn.Sigmoid() # Sigmoid to get channel-wise attention weights
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # Squeeze operation (HxW -> 1x1)
        y = self.fc(y).view(b, c, 1, 1) # Excitation operation (get channel weights)
        return x * y.expand_as(x) # Scale input features by attention weights


class CASModule(nn.Module):
    """
    Collaborative Attention Skip-connection (CAS) Module.
    This module refines encoder features (low-level, skip connection)
    based on upsampled decoder features (high-level) to reduce noise
    and semantic gaps before concatenation.
    """
    def __init__(self, encoder_channels, decoder_channels, reduction=4):
        super().__init__()
        
        self.conv_enc = nn.Conv2d(encoder_channels, decoder_channels, kernel_size=1)
        self.conv_dec = nn.Conv2d(decoder_channels, decoder_channels, kernel_size=1)

        # Squeeze and Excitation Block for channel information interaction on fused features
        self.seb = SEBlock(decoder_channels, reduction=reduction) 
        

        self.spatial_attention_conv_h = nn.Conv2d(decoder_channels, 1, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.spatial_attention_conv_v = nn.Conv2d(decoder_channels, 1, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoder_features, decoder_features_upsampled):
        # 1. Transform encoder and upsampled decoder features
        enc_t = self.conv_enc(encoder_features)
        dec_t = self.conv_dec(decoder_features_upsampled)

        # 2. Fuse features by pixel-wise addition
        fused_features = enc_t + dec_t

        # 3. Process fused features by SEB
        seb_output = self.seb(fused_features)

        # 4. Get pixel-wise weight using 1x3 conv (and 3x1 for symmetric spatial attention)
        spatial_weights = self.spatial_attention_conv_h(seb_output) + self.spatial_attention_conv_v(seb_output)
        attention_map = self.sigmoid(spatial_weights)
        
        # 5. Multiply element-wise with original encoder features (skip connection)
        # This selectively enhances/suppresses parts of the encoder features for the skip path.
        weighted_encoder_features = encoder_features * attention_map.expand_as(encoder_features)
        
        return weighted_encoder_features
