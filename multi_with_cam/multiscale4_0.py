import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ConvNet(nn.Module):
    """Resnet18を使う"""

    def __init__(self):
        """Initialize EmbeddingNet model."""
        super(ConvNet, self).__init__()

        # Everything except the last linear layer
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2]) #7×7の特徴マップ、チャンネル512


    def forward(self, x):
        out = self.features(x)
        return out
    
class Multi(nn.Module):
    def __init__(self):
        super(Multi, self).__init__()

        self.conv_model = ConvNet()  # ResNet18
        
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=512, kernel_size=30, padding=1,
                                     stride=16)  # 1st sub sampling
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=512, kernel_size=30, padding=2,
                                     stride=28)  # 2nd sub sampling
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=6, stride=1, padding=2)
        
        self.fc = torch.nn.Linear(1536, 196)
        
    def forward(self, x):
        conv_input = self.conv_model(x)
        
        first_input = self.conv1(x)
        first_input = self.maxpool1(first_input)

        second_input = self.conv2(x)
        second_input = self.maxpool2(second_input)
        
        merge_triple = torch.cat([conv_input, first_input, second_input], 1)

        merge_gap = F.adaptive_avg_pool2d(merge_triple, (1, 1))

        merge = merge_gap.view(merge_gap.size(0), -1)

        final_input = self.fc(merge)

        return final_input
    
Multi()

"""
from torchsummary import summary
model = Multi()
summary(model,(3,224,224)) # summary(model,(channels,H,W))
"""