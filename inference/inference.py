import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as tfms
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import cv2 
import torch
from PIL import Image
from collections import defaultdict
from tqdm import tqdm_notebook as tqdm
import sys
import copy
import scipy
import time
import tensorflow as tf
import efficientnet.tfkeras as efn
from pympler import asizeof
import vit_keras

from matplotlib.colors import LinearSegmentedColormap
ncolors = 256
color_array = plt.get_cmap('jet')(range(ncolors))
color_array[:,-1] = np.linspace(0.15,1.0,ncolors)
map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha',colors=color_array)
plt.register_cmap(cmap=map_object)

import base64
from pycocotools import _mask as coco_mask
import typing as t
import zlib
import skimage
import pickle

from hpacellseg.cellsegmentator import *
from hpacellseg import cellsegmentator, utils

from config import CFG
from dataset_test import HPADataset_Test
from cell_seg_tools import CellSegmentator, Yield_Images_Dataset
import models_inf

from albumentations.pytorch import ToTensorV2
from albumentations import Compose

np.set_printoptions(precision=4, suppress=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_virtual_device_configuration(
        gpu,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5600)])
    
      #tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


MODEL_PATH = os.getcwd() + "/inference/models-hpa/"

PATH_SCALER_GRADBOOST = os.getcwd() + "/inference/models-hpa/"

MODELS_LIST_EFFNET = [
                      "efficientnet-b4_rgby_lr_0.001_ADAM_steplr_g085_focal1_g1.0_resize640_mediumaug_3.pth",
                      "efficientnet-b4_rgby_lr_0.0015_ADAM_focal1_g1.0_resize640_10pcTest_HEAVY_AUG_E3_F0.pth",
                      "efficientnet-b4_rgby_lr_0.001_ADAM_steplr_g085_focal1_g1.0_resize640_mediumaug_E4_F4.pth",
                      ]
                      
MODELS_LIST_RESNEST = [
                       "resnest101_rgby_lr_0.002_SGD_polyoptim_focal1_g1.0_scheduler_largedataset_resize640_mediumaug_3.pth",
                       "resnest101_rgby_lr_0.002_SGD_focal1_g1.0_resize640_5pcTest_BS8_E3_F0.pth",
                      ]       

MODEL_LABELS_EFF = os.getcwd() + "/inference/models-hpa/model_green.06-0.07.h5"
MODEL_LABELS_RN = os.getcwd() + "/inference/models-hpa/model_rgb_resnext101.09-0.10.h5"
MODEL_LABELS_VIT = os.getcwd() + "/inference/models-hpa/ggg_ViTB16_RedPlat_ADAMW_BCE_EPOCH12-VAL0.0957.h5"


scaler_resnest0 = pickle.load(open(f"{PATH_SCALER_GRADBOOST}/scaler_resnest0_2.pkl", 'rb'))
scaler_resnest1 = pickle.load(open(f"{PATH_SCALER_GRADBOOST}/scaler_resnest1_2.pkl", 'rb'))
scaler_effnet0 = pickle.load(open(f"{PATH_SCALER_GRADBOOST}/scaler_effnet0_2.pkl", 'rb'))
scaler_effnet1 = pickle.load(open(f"{PATH_SCALER_GRADBOOST}/scaler_effnet1_2.pkl", 'rb'))
scaler_effnet2 = pickle.load(open(f"{PATH_SCALER_GRADBOOST}/scaler_effnet2_2.pkl", 'rb'))

model_gradboost_resnest0 =  pickle.load(open(f"{PATH_SCALER_GRADBOOST}/GradientBoostingRegressor_resnest0_2.pkl", 'rb'))
model_gradboost_resnest1 =  pickle.load(open(f"{PATH_SCALER_GRADBOOST}/GradientBoostingRegressor_resnest1_2.pkl", 'rb'))
model_gradboost_effnet0 =  pickle.load(open(f"{PATH_SCALER_GRADBOOST}/GradientBoostingRegressor_effnet0_2.pkl", 'rb'))
model_gradboost_effnet1 =  pickle.load(open(f"{PATH_SCALER_GRADBOOST}/GradientBoostingRegressor_effnet1_2.pkl", 'rb'))
model_gradboost_effnet2 =  pickle.load(open(f"{PATH_SCALER_GRADBOOST}/GradientBoostingRegressor_effnet2_2.pkl", 'rb'))

data_df_sample_submission = pd.read_csv(os.getcwd() + '/inference/test/sample_submission.csv')
data_df = data_df_sample_submission

label_names = [
'0-Nucleoplasm',
'1-Nuclear membrane',
'2-Nucleoli',
'3-Nucleoli fibrillar center',
'4-Nuclear speckles',
'5-Nuclear bodies',
'6-Endoplasmic reticulum',
'7-Golgi apparatus',
'8-Intermediate filaments',
'9-Actin filaments',
'10-Microtubules',
'11-Mitotic spindle',
'12-Centrosome',
'13-Plasma membrane',
'14-Mitochondria',
'15-Aggresome',
'16-Cytosol',
'17-Vesicles + cytosolic patterns',
'18-Negative'
]

def get_transforms(*, data_type):
    if data_type == 'valid':
        return Compose([
            Resize(CFG.size, CFG.size),
            ToTensorV2(),
        ])

    elif data_type == 'test_green_model':
      return Compose([
            Resize(600, 600),
            #ToTensorV2(),
        ])
    elif data_type == 'test_green_model_torch':
      return Compose([
            Resize(CFG.size, CFG.size),
            ToTensorV2(),
        ])

def build_decoder(with_labels=True, target_size=(300, 300), ext='jpg'):
    def decode(path):
        if CFG.color_mode_image_level == "ggg":
            file_bytes = tf.io.read_file(path + "_green.png")
            if ext == 'png':
                img = tf.image.decode_png(file_bytes, channels=3)
            elif ext in ['jpg', 'jpeg']:
                img = tf.image.decode_jpeg(file_bytes, channels=3)
            else:
                raise ValueError("Image extension not supported")

            img = tf.cast(img, tf.float32) / 255.0
            img = tf.image.resize(img, target_size)

            return img
        if CFG.color_mode_image_level == "rgb":
            r = tf.io.read_file(path + "_red.png")
            g = tf.io.read_file(path + "_green.png")
            b = tf.io.read_file(path + "_blue.png")

            red = tf.io.decode_png(r, channels=1)
            blue = tf.io.decode_png(g, channels=1)
            green = tf.io.decode_png(b, channels=1)

            red = tf.image.resize(red, target_size)
            blue = tf.image.resize(blue, target_size)
            green = tf.image.resize(green, target_size)

            img = tf.stack([red, green, blue], axis=-1)
            img = tf.squeeze(img)
            img = tf.image.convert_image_dtype(img, tf.float32) / 255
            return img

    def decode_with_labels(path, label):
        return decode(path), label

    return decode_with_labels if with_labels else decode

def build_augmenter(with_labels=True):
    def augment(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        return img

    def augment_with_labels(img, label):
        return augment(img), label

    return augment_with_labels if with_labels else augment

def build_dataset_tf(paths, labels=None, bsize=32, cache=True,
                      decode_fn=None, augment_fn=None,
                      augment=True, repeat=True, shuffle=1024, img_size=300,
                      cache_dir=""):
        if cache_dir != "" and cache is True:
            os.makedirs(cache_dir, exist_ok=True)

        if decode_fn is None:
            decode_fn = build_decoder(labels is not None, target_size=(img_size, img_size))

        if augment_fn is None:
            augment_fn = build_augmenter(labels is not None)

        AUTO = tf.data.experimental.AUTOTUNE
        slices = paths if labels is None else (paths, labels)

        dset = tf.data.Dataset.from_tensor_slices(slices)
        dset = dset.map(decode_fn, num_parallel_calls=AUTO)
        dset = dset.cache(cache_dir) if cache else dset
        dset = dset.map(augment_fn, num_parallel_calls=AUTO) if augment else dset
        dset = dset.repeat() if repeat else dset
        dset = dset.shuffle(shuffle) if shuffle else dset
        dset = dset.batch(bsize).prefetch(AUTO)

        return dset

test_paths = CFG.PATH_TEST + "/" + data_df['ID']

test_decoder_600 = build_decoder(with_labels=False, target_size=(600, 600))
test_decoder_384 = build_decoder(with_labels=False, target_size=(384, 384))

CFG.color_mode_image_level = "ggg"
dtest_tf_green_600 = build_dataset_tf(
        test_paths, bsize=CFG.batch_size_, repeat=False, 
        shuffle=False, augment=False, cache=False,
        decode_fn=test_decoder_600
    )
CFG.color_mode_image_level = "rgb"
dtest_tf_rgb_600 = build_dataset_tf(
        test_paths, bsize=CFG.batch_size_, repeat=False, 
        shuffle=False, augment=False, cache=False,
        decode_fn=test_decoder_600
    )
CFG.color_mode_image_level = "ggg"
dtest_tf_ggg_384 = build_dataset_tf(
        test_paths, bsize=CFG.batch_size_, repeat=False, 
        shuffle=False, augment=False, cache=False,
        decode_fn=test_decoder_384
    )


IMAGE_SIZES = [1728, 2048, 3072, 4096]
predict_df_1728 = data_df[data_df.ImageWidth==IMAGE_SIZES[0]]
predict_df_2048 = data_df[data_df.ImageWidth==IMAGE_SIZES[1]]
predict_df_3072 = data_df[data_df.ImageWidth==IMAGE_SIZES[2]]
predict_df_4096 = data_df[data_df.ImageWidth==IMAGE_SIZES[3]]

assert len(predict_df_1728) + len(predict_df_2048) + len(predict_df_3072) + len(predict_df_4096) == len(data_df), "IMAGE SIZE DFS DONT MATCH SAMPLE SUBMISSION"

NUC_MODEL = os.getcwd()+"/inference/models_cellseg/dpn_unet_nuclei_v1.pth"
CELL_MODEL = os.getcwd()+"/inference/models_cellseg/dpn_unet_cell_3ch_v1.pth"
segmentator_even_faster = CellSegmentator(
    NUC_MODEL,
    CELL_MODEL,
    device="cuda",
    multi_channel_model=True,
    padding=True,
    return_without_scale_restore=True
)

###############WITH DATALODER################
yield_ims_1728 = Yield_Images_Dataset(predict_df_1728)
yield_ims_2048 = Yield_Images_Dataset(predict_df_2048)
yield_ims_3072 = Yield_Images_Dataset(predict_df_3072)
yield_ims_4096 = Yield_Images_Dataset(predict_df_4096)

dataloader_ims_seg_1728 = DataLoader(yield_ims_1728, batch_size=24, shuffle=False, num_workers=0)
dataloader_ims_seg_2048 = DataLoader(yield_ims_2048, batch_size=12, shuffle=False, num_workers=0)
dataloader_ims_seg_3072 = DataLoader(yield_ims_3072, batch_size=3, shuffle=False, num_workers=0)
dataloader_ims_seg_4096 = DataLoader(yield_ims_4096, batch_size=3, shuffle=False, num_workers=0)

dataloaders_all_sizes = [dataloader_ims_seg_1728, dataloader_ims_seg_2048, dataloader_ims_seg_3072, dataloader_ims_seg_4096]

start_time = time.time()
even_faster_outputs = []
output_ids = []
batch_size = 24
sizes_list = []
im_proc = 0

for i, dataloader_ims_seg in enumerate(dataloaders_all_sizes):
    print(f"GETTING IMAGE SIZES: {IMAGE_SIZES[i]}, BATCHES: {len(dataloader_ims_seg)}")
    print
    for blue_images, ryb_images, sizes, _ids in dataloader_ims_seg:

        print(f"SEGMENT COUNT: {im_proc}")

        blue_batch = blue_images.numpy()
        ryb_batch = ryb_images.numpy()

        #print(blue_batch.shape)
        nuc_segmentations = segmentator_even_faster.pred_nuclei(blue_batch)
        cell_segmentations = segmentator_even_faster.pred_cells(ryb_batch, precombined=True)

        for data_id, nuc_seg, cell_seg, size in zip(_ids, nuc_segmentations, cell_segmentations, sizes):
            _, cell = utils.label_cell(nuc_seg, cell_seg)
            even_faster_outputs.append(np.ubyte(cell))
            output_ids.append(data_id)
            sizes_list.append(size.numpy())
        im_proc += len(_ids)
        #if im_proc > 20:
          #break
    del dataloader_ims_seg
    print(time.time() - start_time)

cell_masks_df = pd.DataFrame(list(zip(output_ids, even_faster_outputs,sizes_list)),
                             columns=["ID", "mask", "ori_size"])

cell_masks_df = cell_masks_df.set_index('ID')
cell_masks_df = cell_masks_df.reindex(index=data_df['ID'])
cell_masks_df = cell_masks_df.reset_index()

#CLEAR MEMORY
del sizes_list
del even_faster_outputs
del output_ids
del segmentator_even_faster
del yield_ims_1728 
del yield_ims_2048 
del yield_ims_3072 
del yield_ims_4096
del dataloader_ims_seg_1728 
del dataloader_ims_seg_2048
del dataloader_ims_seg_3072 
del dataloader_ims_seg_4096
del dataloaders_all_sizes
import gc
import ctypes
libc = ctypes.CDLL("libc.so.6")
libc.malloc_trim(0)
gc.collect()
torch.cuda.empty_cache()
torch.cuda.empty_cache()

X_test = data_df["ID"]
assert list(X_test) == list(cell_masks_df["ID"]), "X_Test and cellmask dont match"

test_data = HPADataset_Test(X_test, path=PATH_TEST, transforms=get_transforms(data_type='valid'), mode="cam")
test_data_green_model = HPADataset_Test(X_test, path=PATH_TEST,transforms=get_transforms(data_type='valid'), mode="green_model")


def swish(x, beta=1.0):
    #https://paperswithcode.com/method/swish
    return x * torch.sigmoid(beta*x)

def get_all_cams(batch_cam_scaled, model, scales, ims_per_batch):
  bs = ims_per_batch
  with torch.no_grad():
        ori_w, ori_h = CFG.size, CFG.size
        strided_up_size = (CFG.size, CFG.size)
        all_scale_cams = torch.from_numpy(np.zeros((bs, len(scales), 19, CFG.size, CFG.size))).cuda()
        all_scale_preds = torch.from_numpy(np.zeros((bs, len(scales), 19))).cuda()
#######do scaling beforehand, do augmenting beforehand?#################
        num_channels = 4
        for i, images in enumerate(batch_cam_scaled):

            with torch.cuda.amp.autocast():
                logits, features = model(images, with_cam=True)
            #print(features.max(), features.min())
            features = swish(features)

            #reshape augmented images to im1, im2, ...
            logits = logits.reshape(bs*4//bs//4, 4, bs, 19).mean(1).view(bs*4//4, 19)
            all_scale_preds[:, i, :] = logits#.cpu()
            
            #deaugment features
            features = torch.cat([features[0:bs], features[bs:bs*2].flip(2), features[bs*2:bs*3].flip(3), torch.flip(features[bs*3:bs*4],(3,2))])

            #reshape augmented features to im1, im2, ...
            size_feats = features.shape[-1]
            features = features.reshape(bs*4//bs//4, 4, bs, 19, size_feats, size_feats).sum(1).view(bs*4//4, 19, size_feats, size_feats)
            
            cams = F.interpolate(features,( CFG.size, CFG.size), mode='bicubic', align_corners=False) #try bicubic here :)


            all_scale_cams[:, i, :, :, :] = cams

        
        all_logits = np.sum(all_scale_preds.detach().cpu().numpy(), axis=1)
        all_cams = np.sum(all_scale_cams.detach().cpu().numpy(), axis=1)
        
        print("CAMS DONE")
  return {"hr_cams": all_cams, "logits" : all_logits}



def sigmoid_factor(x, factor=1, move=0):
  return 1 / (1 + np.exp(-factor*(x-move)))


def get_hrcams_vis(data, show_image, model, model_state, scales, ims_per_batch):
    model.load_state_dict(model_state['model_state_dict'])           
    all_cams_test = get_all_cams(data, model, scales, ims_per_batch)

    pred_label = all_cams_test["logits"]/len(scales) #there was an erroring dividing this by len(scales)+1, maybe it's important later!
    print("---------------")
    sig_labels = sigmoid_factor(pred_label) 
    print(sig_labels)
    #keep value of cams always in range as if there are only 3 scales (because GradBoost was trained on 3!)
    all_hr_cams_test = all_cams_test["hr_cams"]*(3/len(scales))
    all_label_names_map = [label_names[i] for i in range(19)]
    
    if show_image:
      plt.rcParams['figure.figsize'] = [200, 25]
      plt.figure()
      f, axarr = plt.subplots(ims_per_batch, 19)
      for b in range(ims_per_batch):
        for i, cam in enumerate(all_hr_cams_test[b]): 
          axarr[b, i].imshow(data[0][b, 0:3].cpu().permute(1, 2, 0)/255, interpolation="bicubic") #image 0 is non augmented!
          axarr[b, i].set_title(f"{all_label_names_map[i]}::{sig_labels[b][i]}")
          axarr[b, i].imshow(cam.squeeze(), cmap="rainbow_alpha",alpha=0.9)

      plt.show()

    return all_hr_cams_test, sig_labels


def encode_binary_mask(mask: np.ndarray) -> t.Text:
  """Converts a binary mask into OID challenge encoding ascii text."""

  # check input mask --
  if mask.dtype != np.bool:
    raise ValueError(
        "encode_binary_mask expects a binary mask, received dtype == %s" %
        mask.dtype)

  mask = np.squeeze(mask)
  if len(mask.shape) != 2:
    raise ValueError(
        "encode_binary_mask expects a 2d mask, received shape == %s" %
        mask.shape)

  # convert input mask to expected COCO API input --
  mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
  mask_to_encode = mask_to_encode.astype(np.uint8)
  mask_to_encode = np.asfortranarray(mask_to_encode)

  # RLE encode mask --
  encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

  # compress and base64 encoding --
  binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
  base64_str = base64.b64encode(binary_str)
  return base64_str

def get_all_encoded_cells(mask):
  print(mask.shape)
  cell_masks = []
  for i in range(1, np.max(mask)+1):
    enc_mask = encode_binary_mask((mask == i))
    cell_masks.append(enc_mask)
  return cell_masks

def resize_mask(mask):
    resized_mask = resize_full_mask(mask, CFG.size)
    cell_masks = []
    for i in range(1, np.max(mask)+1):
      cell_masks.append((resized_mask == i))

    return cell_masks

def resize_full_mask(mask, size):
    #zoom_factor = size / mask.shape[0]
    #resized_mask = scipy.ndimage.zoom(mask, zoom_factor, order=0) #change zoom method....
    resized_mask = cv2.resize(mask,(size,size),interpolation=cv2.INTER_NEAREST_EXACT)
    return resized_mask

def get_pred_string(mask_probas, cell_masks_fullsize_enc):
  assert len(mask_probas) == len(cell_masks_fullsize_enc), "Probas have different length than masks"
  string = ""
  for enc_mask, mask_proba in zip(cell_masks_fullsize_enc, mask_probas):
      for cls, proba in enumerate(mask_proba):
        #print(cls, proba)
        string += str(cls) + " " + str(proba) + " "  + enc_mask.decode("utf-8") + " "

  return string

def show_seg_cells(data, cell_mask_list, batch_ids, batch_ims):
    plt.rcParams['figure.figsize'] = [20, 10]
    plt.figure()
    f, axarr = plt.subplots(1, len(cell_mask_list))

    cell_mask_list = copy.deepcopy(cell_mask_list)
    for b, cell_mask in enumerate(cell_mask_list):
          cell_mask = copy.deepcopy(cell_mask)

          id = batch_ids[b]
          resized_masks = cell_mask
          resized_image = skimage.transform.resize(batch_ims[b, 0:3].permute(1, 2, 0), (CFG.size,CFG.size), order=1)  #image called for showing

          for label, mask  in enumerate(resized_masks):
              uint_img = np.array(mask*255).astype('uint8')
              M = cv2.moments(uint_img)
              cX = int(M["m10"] / M["m00"])
              cY = int(M["m01"] / M["m00"])
              axarr[b].text(cX, cY, label, fontsize=10,weight='bold', color="white",bbox=dict(facecolor='black', edgecolor='none'))
              axarr[b].contour(mask, 1, colors='cyan', linewidths=1)
          axarr[b].set_title(str(id))
          axarr[b].imshow(resized_image, interpolation="bicubic")
          

    plt.show()



def get_prob_from_cams_masks(cams, masks, labels, labels_from_labelmodel, verbose=True, typ=None):
    
    print(f"GETTING MODEL {typ}")
    if typ=="resnest0":
      scaler = scaler_resnest0
      model_gradboost = model_gradboost_resnest0
    if typ=="resnest1":
      scaler = scaler_resnest1
      model_gradboost = model_gradboost_resnest1
    if typ=="resnest2":
      scaler = scaler_resnest1
      model_gradboost = model_gradboost_resnest1
    if typ=="effnet0":
      scaler = scaler_effnet0
      model_gradboost = model_gradboost_effnet0
    if typ=="effnet1":
      scaler = scaler_effnet1
      model_gradboost = model_gradboost_effnet1
    if typ=="effnet2":
      scaler = scaler_effnet2
      model_gradboost = model_gradboost_effnet2
    if typ=="effnet3":
      scaler = scaler_effnet2
      model_gradboost = model_gradboost_effnet2

    masks_probas = np.zeros((len(masks), 19)) #shape: (n_masks, classes) -> probablities (products of CAM and Cell Mask) for every mask and class
    for i, mask in enumerate(masks):
      for label, cam in enumerate(cams):

        cam_by_mask = np.multiply(mask, cam)
        cam_mask_product = np.multiply(cam_by_mask, labels[label])#**0.7) 
                                                          #normalize different sizes of organellese
        masks_probas[i, label] = np.sum(cam_mask_product)# * size_mask_organelles[label]

      if verbose:
        print(f"MASK: {i} PROB-RAW: {masks_probas[i, :]}") #add 50 to class "negative" ?
        print("--------------------------------------")

    #scaling standardization
    #try sklearn.preprocessing.RobustScaler
    std_scaler = preprocessing.RobustScaler().fit(masks_probas.reshape(-1, 1))
    
    for i, mask in enumerate(masks):
      std_scaled = std_scaler.transform(masks_probas[i, :].reshape(-1, 1))[:,0]
      model_scaled = scaler.transform(masks_probas[i, :].reshape(-1, 1))[:,0]
      if verbose:
        print(f"MASK: {i} STD-Scaled: {std_scaled}")
        print(f"MASK: {i} Model-Scaled: {model_scaled}")
        print("--------------------------------------")

      sigmoid_probas = sigmoid_factor(std_scaled, factor=CFG.sigmoid_factor, move=CFG.sigmoid_move) #put that scaler in (0,1)!
      gradboost_probas = model_gradboost.predict(model_scaled.reshape(-1, 1))
      if verbose:
        print(f"MASK: {i} PROB-SIGMOID: {sigmoid_probas}")
        print(f"MASK: {i} PROB-GradBoost: {gradboost_probas}")
        print("--------------------------------------")

      if CFG.extra_model_for_labels:
        masks_probas[i, :] = sigmoid_probas*CFG.split_sigmoid_graboost[0] + gradboost_probas*CFG.split_sigmoid_graboost[1]
        #weight each final label output with the according label output from the model
        masks_probas[i, :] = CFG.split[0] * masks_probas[i, :] + CFG.split[1] * labels_from_labelmodel + CFG.split[2] * labels

      if verbose:
        print(f"MASK: {i} PROB-WITH-LABELMODEL: {masks_probas[i, :]}")
        #print("LABELS FROM LABEL MODEL WAS")
        #print(labels_from_labelmodel)

    return masks_probas


if CFG.resnest:
    model_resnest =  models_inf.Classifier(CFG.model_name_resnest, CFG.classes, mode="normal")
    if CFG.color_mode == "rgby":
      weight = model_resnest.model.conv1[0].weight.clone()
      model_resnest.model.conv1[0] = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) #64 for resnest101, 32 for resnest50
      with torch.no_grad():
        model_resnest.model.conv1[0].weight[:, :3] = weight
        model_resnest.model.conv1[0].weight[:, 3] = model_resnest.model.conv1[0].weight[:, 0]

    model_resnest.to(device)
    model_resnest.eval()
    model_states_resnet = [torch.load(MODEL_PATH + f"{model}") for model in MODELS_LIST_RESNEST] #do this only once!!!

if CFG.effnet:
    model_effnet = models_inf.Classifier_EffNet(CFG.model_name_effnet)
    if CFG.color_mode == "rgby":
        model_effnet.enet._conv_stem.in_channels = 4
        model_effnet.enet._conv_stem.weight = torch.nn.Parameter(torch.cat([model_effnet.enet._conv_stem.weight, model_effnet.enet._conv_stem.weight[:, 0:1, :, :]], axis=1))
    model_effnet.to(device)
    model_effnet.eval()
    model_states_effnet = [torch.load(MODEL_PATH + f"{model}") for model in MODELS_LIST_EFFNET] #do this only once!!!

if CFG.extra_model_for_labels:
    if CFG.extra_model_is_tf:
      model_green_eff = tf.keras.models.load_model(MODEL_LABELS_EFF)
      model_green_rn = tf.keras.models.load_model(MODEL_LABELS_RN)
      model_green_vit = tf.keras.models.load_model(MODEL_LABELS_VIT, custom_objects={
                                                  'ClassToken': vit_keras.layers.ClassToken, 
                                                  'AddPositionEmbs' : vit_keras.layers.AddPositionEmbs,
                                                  'TransformerBlock' : models_inf.TransformerBlock,
                                                  'MultiHeadSelfAttention' : models_inf.MultiHeadSelfAttention})
      #model_green_dn = tf.keras.models.load_model(MODEL_LABELS_DN)
    else:
      model_for_labels_state = torch.load(MODEL_LABELS)
      model_green = models_inf.Classifier_EffNet_GREEN("efficientnet-b7")
      model_green.to(device)
      model_green.eval()


dl_test = DataLoader(test_data, batch_size=batch_size_, shuffle=False, num_workers=0)



def inference_one_batch(batch_cam, batch_ids_cam, batch_seg, ims_per_batch, batch_eff_tf=None, batch_rn_tf_600=None, batch_vit_tf_384=None, show_image=True, show_seg=True, verbose=True):
    batch_ids_seg = tuple(batch_seg["ID"])
    print(batch_ids_cam)
    print(batch_ids_seg)
    assert batch_ids_cam == batch_ids_seg, "IDS OF SEGMENTATION AND CAMS DONT MATCH"

    #get mask (from fullsize img)
    print(f"GETTING {ims_per_batch} CELL MASKS")
    cell_mask_list = batch_seg["mask"] 
    cell_mask_sizes = batch_seg["ori_size"]
    cell_masks_full_size = []
    #resize masks to original size
    for e, (mask, size) in enumerate(zip(cell_mask_list, cell_mask_sizes)):
      cell_masks_full_size.append(resize_full_mask(mask,size)) #change zoom method...    
    
    #resize masks to image size
    res_cell_masks = [resize_mask(cell_mask) for cell_mask in cell_mask_list]

    if show_seg:
      show_seg_cells(batch_cam, res_cell_masks, batch_ids_seg, batch_cam)

    del cell_mask_list
    #get encoded cell masks -> save
    cell_masks_fullsize_enc_list = []
    for cell_mask in cell_masks_full_size:
        cell_masks_fullsize_enc = get_all_encoded_cells(cell_mask)
        cell_masks_fullsize_enc_list.append(cell_masks_fullsize_enc)
    del cell_masks_full_size

    print(f"ENCODED {len(cell_masks_fullsize_enc_list)} CELL MASKS")


    #batch_eff_tf -> ggg 600
    #batch_rn_tf_600 -> rgb 600
    #batch_vit_tf_384 -> ggg 384
    labels_model_eff = get_separate_labels_tf(batch_eff_tf, model_green_eff, verbose=verbose, name=MODEL_LABELS_EFF)
    labels_model_rn = get_separate_labels_tf(batch_rn_tf_600, model_green_rn, verbose=verbose, name=MODEL_LABELS_RN)
    labels_model_vit = get_separate_labels_tf(batch_vit_tf_384, model_green_vit, verbose=verbose, name=MODEL_LABELS_VIT)

    labels_model = labels_model_eff*CFG.split_image_level[0] + labels_model_rn*CFG.split_image_level[1] + labels_model_vit*CFG.split_image_level[2]

    print("LABELS FROM LABEL MODEL")
    print(labels_model)

    print("RESIZE IMAGES")
    batch_cam_scaled = []
    scales = [1.0, 1.3, 1.6]
    st = time.time()
    for i, scale in enumerate(scales):
      image_batch_pil = torch.from_numpy(np.zeros((ims_per_batch, 4, round(CFG.size*scale), round(CFG.size*scale))))
      image_batch = copy.deepcopy(batch_cam)
      for j, im in enumerate(image_batch):
        im = ToPILImage()(im)
        im = im.resize((round(CFG.size*scale), round(CFG.size*scale)), resample=PIL.Image.BICUBIC)
        im = torchvision.transforms.functional.to_tensor(im)*255 #find a way to convert PIL to tensor without scaling (dividing everything by 255)
        image_batch_pil[j] = im

      image_batch_resized = image_batch_pil.float()
      image_batch_augs_fl2 = image_batch_resized.flip(2)
      image_batch_augs_fl3 = image_batch_resized.flip(3)
      image_batch_augs_fl32 = torch.flip(image_batch_resized,(3,2))

      #concat to one vector: im1, im2, .., im1fl, im2fl, .., im1fl2, im2fl2, ...
      images = torch.cat([image_batch_resized, image_batch_augs_fl2, image_batch_augs_fl3, image_batch_augs_fl32], dim=0)
      images = images.cuda()
      batch_cam_scaled.append(images)
    print(f"TIME FOR RESIZING WITH PIL {time.time() - st}")

    if CFG.resnest:
        #get cams (from resized img)
        print("GETTING CAMS AND PREDS RESNEST")
        #scales = [ 0.7, 0.9, 1.0, 1.3, 1.6]
        #scales = [0.9, 1.0, 1.3]

        
        mask_probas_resnest_folds = []
        folds_resnest = len(MODELS_LIST_RESNEST)
        all_ims_resnest = []
        time_spent_mask_probas = []

        for f, model_state_rn in enumerate(model_states_resnet): #iterate through model states
            all_hr_cams, sig_labels = get_hrcams_vis(batch_cam_scaled, show_image, model_resnest, model_state_rn, scales, ims_per_batch)  #dont get image every time again get at beginning!
            
            mask_probas_resnest_batches = []
            #get probabilities
            print(f"GETTING MASKS PROBAS FOLD {f}")
            for b, (cams, mask, sig_label, label_model) in enumerate(zip(all_hr_cams, res_cell_masks, sig_labels, labels_model)): #iterate through batch
                print(f"GETTING MASKS PROBAS BATCH {b}")
                mask_probas_resnest = get_prob_from_cams_masks(cams, mask, sig_label, label_model, verbose, typ=f"resnest{f}")

                if verbose:
                  print(mask_probas_resnest)
                  
                mask_probas_resnest_batches.append(mask_probas_resnest)
            mask_probas_resnest_folds.append(mask_probas_resnest_batches)

        #mask_probas_resnest_folds[0][0] #image0 fold0
        #mask_probas_resnest_folds[1][0] #image0 fold1
        for b in range(ims_per_batch):
            all_ims_resnest.append(np.mean(np.array([mask_probas_resnest_folds[i][b] for i in range(folds_resnest)]), axis=0))


    if CFG.effnet:
        #get cams (from resized img)
        print("GETTING CAMS AND PREDS EFFICIENTNET")
        #scales = [ 0.7, 0.9, 1.0, 1.3, 1.6]
        #scales =[0.9, 1.0, 1.3]

        
        mask_probas_effnet_folds = []
        folds_effnet = len(MODELS_LIST_EFFNET)
        all_ims_effnet = []
        time_spent_mask_probas = []

        for f, model_state_eff in enumerate(model_states_effnet):
            all_hr_cams, sig_labels = get_hrcams_vis(batch_cam_scaled, show_image, model_effnet, model_state_eff, scales, ims_per_batch) #dont get image every time again get at beginning!
            mask_probas_effnet_batches = []
            #get probabilities
            print(f"GETTING MASKS PROBAS FOLD {f}")
            for b, (cams, mask, sig_label, label_model) in enumerate(zip(all_hr_cams, res_cell_masks, sig_labels, labels_model)):
                print(f"GETTING MASKS PROBAS BATCH {b}")
                s_t = time.time()
                mask_probas_effnet = get_prob_from_cams_masks(cams, mask, sig_label, label_model, verbose, typ=f"effnet{f}")
                
                if verbose:
                  print(mask_probas_effnet)

                mask_probas_effnet_batches.append(mask_probas_effnet)

            mask_probas_effnet_folds.append(mask_probas_effnet_batches)

        for b in range(ims_per_batch):
            all_ims_effnet.append(np.mean(np.array([mask_probas_effnet_folds[i][b] for i in range(folds_effnet)]), axis=0))

    if CFG.resnest and CFG.effnet:
      mask_probas = CFG.split_cam_level[0]*np.array(all_ims_effnet) +  CFG.split_cam_level[1]*np.array(all_ims_resnest)

    elif CFG.resnest and not CFG.effnet:
      mask_probas = all_ims_resnest
      
    elif CFG.effnet and not CFG.resnest:
      mask_probas = all_ims_effnet

    if verbose:
      print(mask_probas)  

    return batch_ids_cam, sizes, cell_masks_fullsize_enc_list, mask_probas


df = pd.DataFrame(columns=["image_id", "pred"])
i = 0

start_time = time.time()
ims_done = 0
for i, ((batch_cam, batch_ids_cam), (batch_tf_green_600), (batch_tf_rgb_600), (batch_tf_ggg_384)) in enumerate(zip(dl_test, dtest_tf_green_600, dtest_tf_rgb_600, dtest_tf_ggg_384)):  #dtest_tf
  ims_per_batch = len(batch_ids_cam)
  batch_seg = cell_masks_df[i*batch_size_ : i*batch_size_  + batch_size_]
  
  ids, sizes, cell_masks_fullsize_enc_list, probas =  inference_one_batch(batch_cam, batch_ids_cam, batch_seg, ims_per_batch, batch_eff_tf=batch_tf_green_600, batch_rn_tf_600=batch_tf_rgb_600, batch_vit_tf_384=batch_tf_ggg_384, show_image=False, show_seg=False, verbose=False)
  for id, proba, cell_mask_enc in zip(ids, probas, cell_masks_fullsize_enc_list):
    pred_string = get_pred_string(proba, cell_mask_enc)
    d = {"image_id":id, "pred": pred_string}
    df = df.append(d, ignore_index=True)
  ims_done += ims_per_batch
  print(f"---{ims_done} IMAGES DONE---")
  
  torch.cuda.empty_cache()
  gc.collect()
print("--- %s seconds ---" % (time.time() - start_time))


sub = sub[data_df.columns]

sub.to_csv("submission.csv",index=False)