import os
import torch
import time
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
from sklearn.model_selection import KFold
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

import tensorflow as tf

from utils_ar import AverageMeter, init_logger
from config import CFG
from augmentations import get_transforms
from dataset_torch import HPADataset
from metrics_ar import calculate_metrics, calculate_roc
from effnet_cam import Classifier_EffNet

sys.path.append("/PuzzleCAM")
from PuzzleCAM.tools.general.io_utils import *
from PuzzleCAM.tools.general.time_utils import *
from PuzzleCAM.tools.general.json_utils import *

from PuzzleCAM.tools.ai.demo_utils import *
from PuzzleCAM.tools.ai.optim_utils import *
from PuzzleCAM.tools.ai.torch_utils import *

from PuzzleCAM.tools.ai.augment_utils import *
from PuzzleCAM.tools.ai.randaugment import *
from PuzzleCAM.core.puzzle_utils import *
from PuzzleCAM.core.networks import *

LOGGER = init_logger()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_df = pd.read_csv(os.getcwd() + "/train_data/train.csv")
public_hpa_df_except16_0 = pd.read_csv(os.getcwd() + '/train_data/hpa_public_excl_0_16_768.csv')

train_df["Labels_list"] = train_df["Label"].str.split("|").apply(lambda x: [int(i) for i in x])
public_hpa_df_except16_0["Labels_list"] = public_hpa_df_except16_0["Label"].str.split("|").apply(lambda x: [int(i) for i in x])
train_kaggle_public = pd.concat([train_df, public_hpa_df_except16_0], ignore_index=True, sort=False)


mlb = MultiLabelBinarizer()

X = train_kaggle_public['ID']
y = train_kaggle_public['Label'].str.split("|").apply(lambda x: [int(i) for i in x])

df_ohe = pd.DataFrame(mlb.fit_transform(y),columns=mlb.classes_)
df_ohe_np = df_ohe.to_numpy()

msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)

for train_index, test_index in msss.split(X, df_ohe_np):
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]

train_data = HPADataset(X_train, y_train, mode="train", tfms=get_transforms(data_type='train'))
test_data = HPADataset(X_test, y_test, mode="test",tfms=get_transforms(data_type='valid'))
full_data = HPADataset(X, y, mode="test", tfms=get_transforms(data_type='valid'))


class FocalLoss(nn.Module):
    def __init__(self, gamma=2): #lower gamma to give the "hard-to-classify" examples less importancy (model is overdonficent for mitotic spindle!), maybe from 2 to 1
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.sum(dim=1).mean()



class PolyOptimizer(torch.optim.SGD):
    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9, nesterov=False, global_step=0):
        super().__init__(params, lr, weight_decay, nesterov=nesterov)

        self.global_step = global_step
        self.max_step = max_step
        self.momentum = momentum
        
        self.__initial_lr = [group['lr'] for group in self.param_groups]
    
    def step(self, closure=None):
        if self.global_step < self.max_step:

            lr_mult = (1 - self.global_step*CFG.gradient_accumulation_steps / self.max_step) ** self.momentum # *CFG.gradient_accumulation_steps
            
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1

def train_fn(model, train_loader, class_loss_fn, optimizer, epoch, gap_fn, re_loss_fn, scheduler=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    start = end = time.time()
    global_step = 0
    class_losses = AverageMeter()
    p_class_losses = AverageMeter()
    re_losses = AverageMeter()
    conf_losses = AverageMeter()
    iters = len(train_loader)
    model.train()

    for step, (images, labels) in enumerate(train_loader):
      batch_size = labels.size(0)
      images, labels = images.to(device), labels.to(device)
      images, labels = images.float(), labels.float()

      with torch.cuda.amp.autocast():
        logits, features = model(images, with_cam=True)

      tiled_images = tile_features(images, CFG.num_pieces)
      with torch.cuda.amp.autocast():
        tiled_logits, tiled_features = model(tiled_images, with_cam=True)

      re_features = merge_features(tiled_features, CFG.num_pieces, CFG.batch_size)

      
      ##### LOSSES #####
      loss_option = CFG.loss_option.split('_')
      with torch.cuda.amp.autocast():
          if CFG.level == 'cam':
              features = make_cam(features)
              re_features = make_cam(re_features)
          
          class_loss = class_loss_fn(logits, labels).mean()

          if 'pcl' in loss_option:
              p_class_loss = class_loss_fn(gap_fn(re_features), labels).mean()
          else:
              p_class_loss = torch.zeros(1).to(device)
          
          if 're' in loss_option:
              if CFG.re_loss_option == 'masking':
                  class_mask = labels.unsqueeze(2).unsqueeze(3)
                  re_loss = re_loss_fn(features, re_features) * class_mask
                  re_loss = re_loss.mean()
              elif CFG.re_loss_option == 'selection':
                  re_loss = 0.
                  for b_index in range(labels.size()[0]):
                      class_indices = labels[b_index].nonzero(as_tuple=True)
                      selected_features = features[b_index][class_indices]
                      selected_re_features = re_features[b_index][class_indices]
                      
                      re_loss_per_feature = re_loss_fn(selected_features, selected_re_features).mean()
                      re_loss += re_loss_per_feature
                  re_loss /= labels.size()[0]
              else:
                  re_loss = re_loss_fn(features, re_features).mean()
          else:
              re_loss = torch.zeros(1).to(device)

          if 'conf' in loss_option:
              conf_loss = shannon_entropy_loss(tiled_logits)
          else:
              conf_loss = torch.zeros(1).to(device)
          
          if CFG.alpha_schedule == 0.0:
              alpha = CFG.alpha
          else:
              alpha = min(CFG.alpha * epoch / (CFG.epochs * CFG.alpha_schedule), CFG.alpha)
      
      loss = class_loss + p_class_loss + alpha * re_loss + conf_loss
      loss_raw = class_loss.item() + p_class_loss.item() + re_loss.item() + conf_loss.item()
      
      loss.mean().backward()
      
      losses.update(loss_raw, batch_size)
      class_losses.update(class_loss.mean().item(), batch_size)
      p_class_losses.update(p_class_loss.mean().item(), batch_size)
      re_losses.update(re_loss.mean().item(), batch_size)
      conf_losses.update(conf_loss.mean().item(), batch_size)

      if (step + 1) % CFG.gradient_accumulation_steps == 0:
          optimizer.step()
          optimizer.zero_grad()
          global_step += 1

      batch_time.update(time.time() - end)
      end = time.time()
      if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
          print('Epoch: [{0}][{1}/{2}] '
                'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                'Elapsed {remain:s} '
                'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                'CLS Loss: ({class_loss.avg:.4f}) '
                'CLS_P Loss: ({p_class_loss.avg:.4f}) '
                'RE Loss: ({re_loss.avg:.4f}) '
                'CONF Loss: ({conf_loss.avg:.4f}) '
                'LR: ({lr:.8f})'
                .format(
                  epoch, step, len(train_loader), batch_time=batch_time,
                  data_time=data_time, 
                  loss=losses,
                  class_loss=class_losses,
                  p_class_loss=p_class_losses,
                  re_loss=re_losses,
                  conf_loss=conf_losses,
                  lr= optimizer.param_groups[0]['lr'],
                  remain=timeSince(start, float(step+1)/len(train_loader)),
                  ))
          
    return losses.avg

def sigmoid_array(x):                                        
    return 1 / (1 + np.exp(-x))

def val_fn(model, val_loader, class_loss_fn, epoch, gap_fn, re_loss_fn):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    start = end = time.time()
    global_step = 0
    class_losses = AverageMeter()
    p_class_losses = AverageMeter()
    re_losses = AverageMeter()
    conf_losses = AverageMeter()
    preds = []
    y_gt = []

    model.eval()
    with torch.no_grad():
        for step, (images, labels) in enumerate(val_loader):
          batch_size = labels.size(0)
          images, labels = images.to(device), labels.to(device)
          images, labels = images.float(), labels.float()

          with torch.cuda.amp.autocast():
            logits, features = model(images, with_cam=True)

          tiled_images = tile_features(images, CFG.num_pieces)

          with torch.cuda.amp.autocast():
            tiled_logits, tiled_features = model(tiled_images, with_cam=True)

          re_features = merge_features(tiled_features, CFG.num_pieces, CFG.batch_size)

          
          ##### LOSSES #####
          loss_option = CFG.loss_option.split('_')

          with torch.cuda.amp.autocast():
              if CFG.level == 'cam':
                  features = make_cam(features)
                  re_features = make_cam(re_features)
              
              class_loss = class_loss_fn(logits, labels).mean()

              if 'pcl' in loss_option:
                  p_class_loss = class_loss_fn(gap_fn(re_features), labels).mean()
              else:
                  p_class_loss = torch.zeros(1).to(device)
              
              if 're' in loss_option:
                  if CFG.re_loss_option == 'masking':
                      class_mask = labels.unsqueeze(2).unsqueeze(3)
                      re_loss = re_loss_fn(features, re_features) * class_mask
                      re_loss = re_loss.mean()
                  elif CFG.re_loss_option == 'selection':
                      re_loss = 0.
                      for b_index in range(labels.size()[0]):
                          class_indices = labels[b_index].nonzero(as_tuple=True)
                          selected_features = features[b_index][class_indices]
                          selected_re_features = re_features[b_index][class_indices]
                          
                          re_loss_per_feature = re_loss_fn(selected_features, selected_re_features).mean()
                          re_loss += re_loss_per_feature
                      re_loss /= labels.size()[0]
                  else:
                      re_loss = re_loss_fn(features, re_features).mean()
              else:
                  re_loss = torch.zeros(1).to(device)

              if 'conf' in loss_option:
                  conf_loss = shannon_entropy_loss(tiled_logits)
              else:
                  conf_loss = torch.zeros(1).to(device)
              
              if CFG.alpha_schedule == 0.0:
                  alpha = CFG.alpha
              else:
                  alpha = min(CFG.alpha * epoch / (CFG.epochs * CFG.alpha_schedule), CFG.alpha)
          
          loss_raw = class_loss.item() + p_class_loss.item() + re_loss.item() + conf_loss.item()

          losses.update(loss_raw, batch_size)
          class_losses.update(class_loss.mean().item(), batch_size)
          p_class_losses.update(p_class_loss.mean().item(), batch_size)
          re_losses.update(re_loss.mean().item(), batch_size)
          conf_losses.update(conf_loss.mean().item(), batch_size)

          # record accuracy
          preds.append(sigmoid_array(logits.detach().cpu().numpy()))
          y_gt.append(labels.detach().cpu().numpy())

          if CFG.gradient_accumulation_steps > 1:
                loss_raw = loss_raw / CFG.gradient_accumulation_steps

          batch_time.update(time.time() - end)
          end = time.time()
          if step % CFG.print_freq == 0 or step == (len(val_loader)-1):
              print('VAL Epoch: [{0}][{1}/{2}] '
                    'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'Elapsed {remain:s} '
                    'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                    'CLS Loss: ({class_loss.avg:.4f}) '
                    'CLS_P Loss: ({p_class_loss.avg:.4f}) '
                    'RE Loss: ({re_loss.avg:.4f}) '
                    'CONF Loss: ({conf_loss.avg:.4f})'
                    .format(
                      epoch, step, len(val_loader), batch_time=batch_time,
                      data_time=data_time, 
                      loss=losses,
                      class_loss=class_losses,
                      p_class_loss=p_class_losses,
                      re_loss=re_losses,
                      conf_loss=conf_losses,
                      remain=timeSince(start, float(step+1)/len(val_loader)),
                      ))
          
    predictions = np.concatenate(preds)
    y_gts =  np.concatenate(y_gt)
    return losses.avg, predictions, y_gts




def train_loop(train_loader, valid_loader, fold):
  epoch = 0

  if "efficientnet" in CFG.model_name:
    model = Classifier_EffNet(CFG.model_name)
    if CFG.color_mode == "rgby":
      model.enet._conv_stem.in_channels = 4
      model.enet._conv_stem.weight = torch.nn.Parameter(torch.cat([model.enet._conv_stem.weight, model.enet._conv_stem.weight[:, 0:1, :, :]], axis=1))
    if CFG.color_mode == "g":
      model.enet._conv_stem.in_channels = 1
      model.enet._conv_stem.weight = torch.nn.Parameter(model.enet._conv_stem.weight[:, 0:1, :, :])

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.85, last_epoch=-1, verbose=False)


  if "resnest" in CFG.model_name:
    model = Classifier(CFG.model_name, CFG.classes, mode="normal")
    if CFG.color_mode == "rgby": #add 4th channel here
      weight = model.model.conv1[0].weight.clone()
      model.model.conv1[0] = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) #64 for resnest101, 32 for resnest50
      with torch.no_grad():
        model.model.conv1[0].weight[:, :3] = weight
        model.model.conv1[0].weight[:, 3] = model.model.conv1[0].weight[:, 0]


    model.to(device)
    param_groups = model.get_parameter_groups(print_fn=None)
    optimizer = PolyOptimizer([
          {'params': param_groups[0], 'lr': CFG.lr, 'weight_decay': CFG.weight_decay},
          {'params': param_groups[1], 'lr': 2*CFG.lr, 'weight_decay': 0},
          {'params': param_groups[2], 'lr': 10*CFG.lr, 'weight_decay': CFG.weight_decay},
          {'params': param_groups[3], 'lr': 20*CFG.lr, 'weight_decay': 0},
      ], lr=CFG.lr, momentum=0.9, weight_decay=CFG.weight_decay, max_step=CFG.epochs*len(train_loader), nesterov=CFG.nesterov, global_step=epoch*len(train_loader)/CFG.gradient_accumulation_steps)
    scheduler = None
 


  gap_fn = model.global_average_pooling_2d

  class_loss_fn = FocalLoss(gamma=CFG.focal_gamma) #train with high LR and gamma=1.5
  class_loss_fn.to(device)

  re_loss_fn = L1_Loss
  
  if CFG.resume:
      checkpoint = torch.load(CFG.MODEL_PATH)
      epoch = checkpoint['epoch']+1
      model.load_state_dict(checkpoint['model_state_dict'])
      LOGGER.info(f"LOADED MODEL STATES FROM {CFG.MODEL_PATH} EPOCH {epoch}")
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler'])

  best_val_loss = 1000
  best_train_loss = 1000

  for epoch in range(epoch, CFG.epochs):
    start_time = time.time()
    
    avg_loss_train = train_fn(model, train_loader, class_loss_fn, optimizer, epoch, gap_fn, re_loss_fn, scheduler=scheduler)
    avg_loss_val, preds, y_gt = val_fn(model, valid_loader, class_loss_fn, epoch, gap_fn, re_loss_fn)
    if scheduler:
        scheduler.step() #for steplr
    try:
      f1_macros, f1_micros, accuracies = calculate_metrics(preds, y_gt)
      roc_auc = calculate_roc(preds, y_gt)
      LOGGER.info(f"Epoch {epoch} - F1-Macro-Mean: {np.mean(list(f1_macros.values()))} ---- F1-Micro-Mean: {np.mean(list(f1_micros.values()))} ---- Accuracy-Mean: {np.mean(list(accuracies.values()))} ")
      LOGGER.info(f"F1-Macro-0.3: {f1_macros[0.3]} ---- F1-Micro-0.3: {f1_micros[0.3]} ---- Accuracy-0.3: {accuracies[0.3]} ")
      LOGGER.info(f"ROC_AUC: {roc_auc}")
      print(f"F1-Macros: {f1_macros}")
      print(f"F1-Micros: {f1_micros}")
      print(f"Accuracies: {accuracies}")
      
    except:
      LOGGER.info(f"Epoch {epoch} - METRICS COULD NOT BE CALCULATED")

    elapsed = time.time() - start_time
    
    LOGGER.info(f'Epoch {epoch} - avg_train_loss: {avg_loss_train:.4f}  avg_val_loss: {avg_loss_val:.4f} time: {elapsed:.0f}s')

    
    if avg_loss_val < best_val_loss or epoch % 5 == 0:
      
      if avg_loss_val < best_val_loss:
        best_val_loss = avg_loss_val
        LOGGER.info(f'Epoch {epoch} - SAVE NEW BEST MODEL')
      else: 
        LOGGER.info(f'Epoch {epoch} - SAVE MODEL BECAUSE EPOCH')
      try:
        torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'avg_val_loss' : avg_loss_train,
              #'scheduler': scheduler.state_dict(),
              }, CFG.OUTPUT_DIR_MODEL+f'{CFG.experiment_name}_E{epoch}_F{fold}.pth')
      except:
        print("CANNOT SAVE")
    
def main():
    train_loader = DataLoader(train_data, batch_size=CFG.batch_size, num_workers=CFG.num_workers, pin_memory=True, shuffle=True, drop_last=True)
    valid_loader = DataLoader(test_data, batch_size=CFG.batch_size, num_workers=CFG.num_workers, pin_memory=True, shuffle=False, drop_last=True)
    train_loop(train_loader, valid_loader, 0)

if __name__ == '__main__':
    main()