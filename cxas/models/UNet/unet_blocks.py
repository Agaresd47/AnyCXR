import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as parametrizations 

# You might need to add nn.utils.weight_norm for proper weight standardization.
# If you prefer a full custom WeightStandardization layer, you'd define it here.
# For this example, I'll integrate nn.utils.weight_norm as it's built-in.


class DoubleConv(nn.Module):
    """Double Convolution Block: (convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
        Initialize DoubleConv module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            mid_channels (int, optional): Number of mid channels. Defaults to None.
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass of DoubleConv module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.double_conv(x)


class OutConv(nn.Module):
    """Output Convolution Block.

    This block consists of a single convolutional layer without activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """Forward pass through the output convolution block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.conv(x)


# --- New ResidualConvBlock with optional Weight Standardization ---
class ResidualConvBlock(nn.Module):
    """
    Residual Convolutional Block as described in the paper's decoder blocks.
    It applies (Conv => BN => ReLU) * 2 and adds a residual connection.
    Optionally applies Weight Standardization to convolutional layers.
    """
    def __init__(self, in_channels, out_channels, use_ws=True):
        super().__init__()
        
        # Helper to create a Conv2d layer with optional Weight Standardization
        def conv_layer(in_c, out_c, kernel_s, padding_s, use_ws_flag):
            conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_s, padding=padding_s, bias=False)
            if use_ws_flag:
                return parametrizations.weight_norm(conv, name='weight') # name='weight' is often the default
            return conv

        self.block = nn.Sequential(
            # FIX: Pass kernel_s and padding_s by keyword argument
            conv_layer(in_c=in_channels, out_c=out_channels, kernel_s=3, padding_s=1, use_ws_flag=use_ws),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # FIX: Pass kernel_s and padding_s by keyword argument
            conv_layer(in_c=out_channels, out_c=out_channels, kernel_s=3, padding_s=1, use_ws_flag=use_ws),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual connection: 1x1 conv if channel sizes differ, else identity
        self.residual_connection = nn.Identity()
        if in_channels != out_channels:
            # FIX: Pass kernel_s and padding_s by keyword argument
            self.residual_connection = conv_layer(in_c=in_channels, out_c=out_channels, kernel_s=1, padding_s=0, use_ws_flag=use_ws)
            # BN and ReLU are typically applied after the addition for post-activation residuals
            # or before for pre-activation residuals. For simplicity, here we'll assume post-add ReLU.

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.residual_connection(x)
        out = self.block(x)
        return self.relu(out + residual)