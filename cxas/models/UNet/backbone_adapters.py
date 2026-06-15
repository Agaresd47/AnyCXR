import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from collections import OrderedDict

def get_resnet_backbone(model_name: str, pretrained: bool = True):
        """ResNet encoder adapted for 1-channel input"""
        model = getattr(models, model_name)(weights=ResNet50_Weights.DEFAULT)
        
        # Modify first conv layer for 1-channel input
        original_conv1 = model.conv1
        new_conv1 = nn.Conv2d(
            1, 
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )
        
        # Initialize with mean of pretrained RGB weights
        with torch.no_grad():
            new_conv1.weight[:, 0] = original_conv1.weight.mean(dim=1)
        
        return nn.Sequential(OrderedDict([
            ('conv1', nn.Sequential(new_conv1, model.bn1, model.relu)),
            ('maxpool', model.maxpool),
            ('layer1', model.layer1),  # 1/4
            ('layer2', model.layer2),  # 1/8
            ('layer3', model.layer3),  # 1/16
            ('layer4', model.layer4),  # 1/32
        ]))
    
def get_vgg_backbone (self):
    """VGG encoder adapted for 1-channel input"""
    vgg = models.vgg16(pretrained=True).features
    vgg_layers = list(vgg.children())
    
    # Get parameters from first conv layer
    original_conv0 = vgg_layers[0]
    out_channels = original_conv0.out_channels  # Get int value
    kernel_size = original_conv0.kernel_size[0]  # Get kernel size as int (assuming square kernel)
    stride = original_conv0.stride[0]  # Get stride as int (assuming equal strides)
    padding = original_conv0.padding[0]  # Get padding as int (assuming equal padding)
    
    # Create new 1-channel conv layer
    new_conv0 = nn.Conv2d(
        1, 
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )
    
    # Initialize with mean of pretrained RGB weights
    with torch.no_grad():
        new_conv0.weight[:, 0] = original_conv0.weight.mean(dim=1)
    vgg_layers[0] = new_conv0
    
    blocks = [
        nn.Sequential(*vgg_layers[0:4]),   # 1/2
        nn.Sequential(*vgg_layers[4:9]),   # 1/4
        nn.Sequential(*vgg_layers[9:16]),  # 1/8
        nn.Sequential(*vgg_layers[16:23]), # 1/16
        nn.Sequential(*vgg_layers[23:30])  # 1/32
    ]
    
    return nn.Sequential(OrderedDict([
        (f'block{i+1}', block) for i, block in enumerate(blocks)
    ]))