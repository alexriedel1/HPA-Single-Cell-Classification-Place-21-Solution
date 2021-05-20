import sys
sys.path.append("/PuzzleCAM")


from efficientnet_pytorch import EfficientNet
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
import torch.utils.model_zoo as model_zoo

from PuzzleCAM.core.arch_resnet import resnet
from PuzzleCAM.core.arch_resnest import resnest
from PuzzleCAM.core.abc_modules import ABC_Model

from PuzzleCAM.core.deeplab_utils import ASPP, Decoder
from PuzzleCAM.core.aff_utils import PathIndex
from PuzzleCAM.core.puzzle_utils import tile_features, merge_features

from PuzzleCAM.tools.ai.torch_utils import resize_for_tensors

from vit_keras import vit, utils, layers
import tensorflow as tf

class Classifier_EffNet(nn.Module, ABC_Model):
    def __init__(self, backbone, num_classes=19):
        super(Classifier_EffNet, self).__init__()
        self.enet = EfficientNet.from_name(backbone, num_classes=num_classes, in_channels=3, include_top=False)

        dict_sizes = {
            'efficientnet-b0' : 1280,
            'efficientnet-b1' : 1280,
            'efficientnet-b2' : 1408,
            'efficientnet-b3' : 1536,
            'efficientnet-b4' : 1792,
            'efficientnet-b5' : 2048
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
            
class Classifier_EffNet_GREEN(nn.Module, ABC_Model):
    def __init__(self, backbone, num_classes=19):
        super(Classifier_EffNet_GREEN, self).__init__()
        self.enet = EfficientNet.from_name(backbone, num_classes=num_classes, in_channels=3, include_top=False)

        dict_sizes = {
            'efficientnet-b0' : 1280,
            'efficientnet-b1' : 1280,
            'efficientnet-b2' : 1408,
            'efficientnet-b3' : 1536,
            'efficientnet-b4' : 1792,
            'efficientnet-b5' : 2048,
            'efficientnet-b7' : 2560,
            }

        self.dense = nn.Linear(dict_sizes[backbone],19)
        #self.sigmoid = nn.Sigmoid()
        self.initialize([self.dense])

    def forward(self, x, with_cam=False):
        x = self.enet.extract_features(x)
        x = self.global_average_pooling_2d(x)
        logits = self.dense(x)
        #logits = self.sigmoid(logits)
        return logits



#######################################################################
# Normalization
#######################################################################
#from core.sync_batchnorm.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class FixedBatchNorm(nn.BatchNorm2d):
    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps)

def group_norm(features):
    return nn.GroupNorm(4, features)
#######################################################################

class Backbone(nn.Module, ABC_Model):
    def __init__(self, model_name, num_classes=20, mode='fix', segmentation=False):
        super().__init__()

        self.mode = mode

        if self.mode == 'fix': 
            self.norm_fn = FixedBatchNorm
        else:
            self.norm_fn = nn.BatchNorm2d
        
        if 'resnet' in model_name:
            self.model = resnet.ResNet(resnet.Bottleneck, resnet.layers_dic[model_name], strides=(2, 2, 2, 1), batch_norm_fn=self.norm_fn)

            state_dict = model_zoo.load_url(resnet.urls_dic[model_name])
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')

            self.model.load_state_dict(state_dict)
        else:
            if segmentation:
                dilation, dilated = 4, True
            else:
                dilation, dilated = 2, False

            self.model = eval("resnest." + model_name)(pretrained=False, dilated=dilated, dilation=dilation, norm_layer=self.norm_fn)

            del self.model.avgpool
            del self.model.fc

        self.stage1 = nn.Sequential(self.model.conv1, 
                                    self.model.bn1, 
                                    self.model.relu, 
                                    self.model.maxpool)
        self.stage2 = nn.Sequential(self.model.layer1)
        self.stage3 = nn.Sequential(self.model.layer2)
        self.stage4 = nn.Sequential(self.model.layer3)
        self.stage5 = nn.Sequential(self.model.layer4)

class Classifier(Backbone):
    def __init__(self, model_name, num_classes=20, mode='fix'):
        super().__init__(model_name, num_classes, mode)
        
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
        self.num_classes = num_classes

        self.initialize([self.classifier])
    
    def forward(self, x, with_cam=False):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        
        if with_cam:
            features = self.classifier(x)
            logits = self.global_average_pooling_2d(features)
            return logits, features
        else:
            x = self.global_average_pooling_2d(x, keepdims=True) 
            logits = self.classifier(x).view(-1, self.num_classes)
            return logits




CONFIG_B = {
    "dropout": 0.1,
    "mlp_dim": 3072,
    "num_heads": 12,
    "num_layers": 12,
    "hidden_size": 768,
}

class TransformerBlock(tf.keras.layers.Layer):
    """Implements a Transformer block."""

    def __init__(self, *args, num_heads=12, mlp_dim=3072, dropout=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

    def build(self, input_shape):
        self.att = MultiHeadSelfAttention(
            num_heads=self.num_heads,
            name="MultiHeadDotProductAttention_1",
        )
        self.mlpblock = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.mlp_dim,
                    activation="linear",
                    name=f"{self.name}/Dense_0",
                ),
                tf.keras.layers.Lambda(
                    lambda x: tf.keras.activations.gelu(x, approximate=False)
                )
                if hasattr(tf.keras.activations, "gelu")
                else tf.keras.layers.Lambda(
                    lambda x: tfa.activations.gelu(x, approximate=False)
                ),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Dense(input_shape[-1], name=f"{self.name}/Dense_1"),
                tf.keras.layers.Dropout(self.dropout),
            ],
            name="MlpBlock_3",
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_0"
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_2"
        )
        self.dropout = tf.keras.layers.Dropout(self.dropout)

    def call(self, inputs, training):
        x = self.layernorm1(inputs)
        x, weights = self.att(x)
        x = self.dropout(x, training=training)
        x = x + inputs
        y = self.layernorm2(x)
        y = self.mlpblock(y)
        return x + y, weights

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, *args, num_heads=12, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        num_heads = self.num_heads
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {hidden_size} should be divisible by number of heads = {num_heads}"
            )
        self.hidden_size = hidden_size
        self.projection_dim = hidden_size // num_heads
        self.query_dense = tf.keras.layers.Dense(hidden_size, name="query")
        self.key_dense = tf.keras.layers.Dense(hidden_size, name="key")
        self.value_dense = tf.keras.layers.Dense(hidden_size, name="value")
        self.combine_heads = tf.keras.layers.Dense(hidden_size, name="out")

    # pylint: disable=no-self-use
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], score.dtype)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.hidden_size))
        output = self.combine_heads(concat_attention)
        return output, weights