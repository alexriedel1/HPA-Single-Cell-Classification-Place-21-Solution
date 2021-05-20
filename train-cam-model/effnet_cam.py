import sys
sys.path.append("/PuzzleCAM")
from PuzzleCAM.core.abc_modules import ABC_Model
from efficientnet_pytorch import EfficientNet
import torch.nn as nn

class Classifier_EffNet(nn.Module, ABC_Model):
    def __init__(self, backbone, num_classes=19):
        super(Classifier_EffNet, self).__init__()
        self.enet = EfficientNet.from_pretrained(backbone, num_classes=num_classes, in_channels=3, include_top=False)

        dict_sizes = {
            'efficientnet-b0' : 1280,
            'efficientnet-b1' : 1280,
            'efficientnet-b2' : 1408,
            'efficientnet-b3' : 1536,
            'efficientnet-b4' : 1792,
            'efficientnet-b5' : 2048,
            'efficientnet-b6' : 2304,
            'efficientnet-b7' : 2560,
            }
        size_conv2d = dict_sizes[backbone]

        self.classifier = nn.Conv2d(size_conv2d, num_classes, 1, bias=False)
        self.num_classes = num_classes

        self.initialize([self.classifier])
    
    def forward(self, x, with_cam=False):

        x = self.enet.extract_features(x)
       
        if with_cam:
            features = self.classifier(x)
            logits = self.global_average_pooling_2d(features)
            return logits, features
        else:
            x = self.global_average_pooling_2d(x, keepdims=True) 
            logits = self.classifier(x).view(-1, self.num_classes)
            return logits