#import albumentations.pytorch
import albumentations
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip, 
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout, 
    IAAAdditiveGaussianNoise, Transpose, HueSaturationValue, CoarseDropout, ElasticTransform 
    )
from config import CFG
dataset_mean = [0.0994, 0.0466, 0.0606, 0.0879]
dataset_std = [0.1406, 0.0724, 0.1541, 0.1264]

def get_transforms(*, data_type):
    if data_type == "light_train":
      return Compose([
            Resize(CFG.size, CFG.size),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(scale_limit=(0, 0), p=0.5),
            ToTensorV2(),
        ])

    if data_type == "train":
        return Compose([
            Resize(CFG.size, CFG.size),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),    
            albumentations.OneOf([
        albumentations.ElasticTransform(alpha=1, sigma=20, alpha_affine=10),
        albumentations.GridDistortion(num_steps=6, distort_limit=0.1),
        albumentations.OpticalDistortion(distort_limit=0.05, shift_limit=0.05),
    ], p=0.2), 
    albumentations.core.composition.PerChannel(
        albumentations.OneOf([
            albumentations.MotionBlur(p=.05),
            albumentations.MedianBlur(blur_limit=3, p=.05),
            albumentations.Blur(blur_limit=3, p=.05),])
        , p=1.0),
    albumentations.OneOf([
        albumentations.CoarseDropout(max_holes=16, max_height=CFG.size//16, max_width=CFG.size//16, fill_value=0, p=0.5),
        albumentations.GridDropout(ratio=0.09, p=0.5),
        albumentations.Cutout(num_holes=8, max_h_size=CFG.size//16, max_w_size=CFG.size//16, p=0.2),
    ], p=0.5), 
    albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
    ToTensorV2(),
    ],
    additional_targets={
        'r': 'image',
        'g': 'image',
        'b': 'image',
        'y': 'image',
    }
        )
        
    elif data_type == 'valid':
        return Compose([
            Resize(CFG.size, CFG.size),
            ToTensorV2(),
        ])