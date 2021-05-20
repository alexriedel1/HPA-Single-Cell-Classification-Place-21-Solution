from config import CFG
import numpy as np
import cv2 
import os 
from torch.utils.data import Dataset

def load_RGBY_image(image_id, image_size=None):
    red = read_img(image_id, "red", image_size)
    green = read_img(image_id, "green", image_size)
    blue = read_img(image_id, "blue", image_size)
    yellow = read_img(image_id, "yellow", image_size)
    #stacked_images = np.array([red, green, blue])
    if CFG.color_mode == "rgby":
      stacked_images = np.transpose(np.array([red, green, blue, yellow]), (1,2,0))

    if CFG.color_mode == "g":
      stacked_images = np.transpose(np.array([green]), (1,2,0))
      
    if CFG.color_mode == "ggg":
      stacked_images = np.transpose(np.array([green, green, green]), (1,2,0))

    if CFG.color_mode == "rgb":
      stacked_images = np.transpose(np.array([red, green, blue]), (1,2,0))
      
    return stacked_images
  
def read_img(image_id, color, image_size=None):
    filename = f'{CFG.PATH_TRAIN}/{image_id}_{color}.png'
    assert os.path.exists(filename), f'not found {filename}'
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if img.max() > 255:
        img_max = img.max()
        img = (img/255).astype('uint8') ##################SCALED IMAGES HERE###########################
    return img

def one_hot_embedding(label, classes):
   
    vector = np.zeros((classes), dtype = np.float32)
    if len(label) > 0:
        vector[label] = 1.
    return vector
  
class HPADataset(Dataset):
    def __init__(self, ids, labels, tfms=None, mode="train"):
      self.ids = ids
      self.labels = labels
      self.tfms = tfms
      self.mode = mode
    
    def __len__(self):
      return len(self.ids)
    
    def __getitem__(self, idx):
      _ids = self.ids.iloc[idx]
      image = load_RGBY_image(_ids)

      label = self.labels.iloc[idx] #-> one hote encode for puzzle cam!
      y = one_hot_embedding(label, CFG.classes) #18 classes + negative

      if self.tfms:
        augmented = self.tfms(image=image)
        image = augmented['image']

      return image, y