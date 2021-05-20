from config import CFG
import numpy as np
import cv2
import os
from torch.utils.data import Dataset
import tensorflow as tf

def load_RGBY_image(image_id, path, mode="cam", image_size=None): 
    if mode == "green_model":
      green = read_img_scale255(image_id,  "green",path, image_size)
      stacked_images = np.transpose(np.array([green, green, green]), (1,2,0))
      return stacked_images

    if mode=="cam":
      red = read_img(image_id, "red", path, image_size)
      green = read_img(image_id,  "green",path, image_size)
      blue = read_img(image_id,  "blue",path, image_size)
      yellow = read_img(image_id,  "yellow",path, image_size)
    
      if CFG.color_mode == "rgby":
        stacked_images = np.transpose(np.array([red, green, blue,yellow]), (1,2,0))
      else:
        stacked_images = np.transpose(np.array([red, green, blue]), (1,2,0))
      return stacked_images

  
def read_img(image_id, color, path, image_size=None):
    filename = f'{path}/{image_id}_{color}.png'
    assert os.path.exists(filename), f'not found {filename}'
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if image_size is not None:
        img = cv2.resize(img, (image_size, image_size))
    
    if img.max() > 255:
        img_max = img.max()
        img = (img/255).astype('uint8')
    
    return img

def read_img_scale255(image_id, color, path, image_size=None):
    filename = f'{path}/{image_id}_{color}.png'
    assert os.path.exists(filename), f'not found {filename}'
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if image_size is not None:
        img = cv2.resize(img, (image_size, image_size))
    if img.max() > 255:
        img_max = img.max()
    img = (img/255).astype('uint8')/255
    
    return img

def one_hot_embedding(label, classes):
    vector = np.zeros((classes), dtype = np.float32)
    if len(label) > 0:
        vector[label] = 1.
    return vector

class HPADataset_Test(Dataset):
    def __init__(self, ids, path=None, transforms=None, mode="cam"):
      self.ids = ids
      self.transforms = transforms
      self.mode = mode
      self.path = path
    
    def __len__(self):
      return len(self.ids)
    
    def __getitem__(self, idx):
      _ids = self.ids.iloc[idx]
      image = load_RGBY_image(_ids, self.path, self.mode)

      if self.transforms:
        augmented = self.transforms(image=image)
        image = augmented['image']
        #image = tfms.ToPILImage()(image)

      return image, _ids

