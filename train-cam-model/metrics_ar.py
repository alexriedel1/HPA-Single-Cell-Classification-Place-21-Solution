import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
import tensorflow as tf

def calculate_metrics(preds, gt):
  threshs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
  f1_macros  = {0.1 : 0, 0.2 : 0, 0.3 : 0, 0.4 : 0, 0.5 : 0, 0.6 : 0, 0.7 : 0}
  f1_micros  = {0.1 : 0, 0.2 : 0, 0.3 : 0, 0.4 : 0, 0.5 : 0, 0.6 : 0, 0.7 : 0}
  accuracies = {0.1 : 0, 0.2 : 0, 0.3 : 0, 0.4 : 0, 0.5 : 0, 0.6 : 0, 0.7 : 0}

  for thresh in threshs:
    try:
      preds_thresh = np.where(preds > thresh) #calculated with many thresholds...

      preds_df = pd.DataFrame({"id" : preds_thresh[0], "pred": preds_thresh[1]})
      preds_list = preds_df.groupby('id')["pred"].apply(list).reset_index(name='preds')

      preds_binary = MultiLabelBinarizer().fit_transform(preds_list["preds"][:9500]) #no label 11 in test data!
      gt = gt[:9500]
      if preds_binary.shape[1] < 19:
        print(f"Only {preds_binary.shape[1]} labels found...")
        for _ in range(19 - preds_binary.shape[1]):
          preds_binary = np.insert(preds_binary, 0, 0., axis=1) #if threshold is too high, one class can be missing (or even more), so add if missing

      f1_macro = metrics.f1_score(preds_binary, gt, average="macro") #Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
      f1_micro = metrics.f1_score(preds_binary, gt, average="micro") #Calculate metrics globally by counting the total true positives, false negatives and false positives.
      accuracy = metrics.accuracy_score(preds_binary, gt) #In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
      print(f1_macro, f1_micro, accuracy)
      f1_macros[thresh] = f1_macro
      f1_micros[thresh] = f1_micro
      accuracies[thresh] = accuracy
    except:
      print(f"Threshold {thresh} could not be calculated")

  return f1_macros, f1_micros, accuracies

def calculate_roc(preds, gt):
  with tf.device('/cpu:0'):
      m = tf.keras.metrics.AUC(multi_label=False)
      m.update_state(gt[:9500], preds[:9500])
      res = m.result().numpy()
  return res