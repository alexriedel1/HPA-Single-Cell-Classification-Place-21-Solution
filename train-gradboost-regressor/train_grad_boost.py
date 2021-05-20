from ast import literal_eval
import ast
import re
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import sklearn
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor,GradientBoostingRegressor, HistGradientBoostingRegressor,ExtraTreesRegressor,GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import (
    LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor)
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
from pickle import dump
import pickle
from sklearn import metrics
import os
checkpoint = pd.read_pickle(os.getcwd() + "/MASK_CAMS_DATA7200.csv")


def get_cam_mask_label(idx_fold, df, col_name):
    cam_mask_label = []
    y_true = []
    for i in range(len(df)):
      for mask in checkpoint[col_name][i][idx_fold]:
        for n, c in enumerate(mask):

            cam_mask_label.append(c)
            y_true.append(checkpoint["y_true"][i][n])

    return np.hstack(cam_mask_label).reshape(-1, 1), np.hstack(y_true).reshape(-1, 1)


cam_mask_label_resnest_0, y_true_resnest_0 = get_cam_mask_label(0, checkpoint, "cam_intensities_per_class_resnest")
cam_mask_label_resnest_1, y_true_resnest_1 = get_cam_mask_label(1, checkpoint, "cam_intensities_per_class_resnest")
cam_mask_label_effnet_0, y_true_effnet_0 =get_cam_mask_label(0, checkpoint, "cam_intensities_per_class_effnet")
cam_mask_label_effnet_1, y_true_effnet_1 = get_cam_mask_label(1, checkpoint, "cam_intensities_per_class_effnet")
cam_mask_label_effnet_2, y_true_effnet_2 = get_cam_mask_label(2, checkpoint, "cam_intensities_per_class_effnet")

def save_scaler_model(scaler, model, name):
    dump(scaler, open(os.getcwd() + f'/scaler_{name}.pkl', 'wb'))
    dump(model, open(os.getcwd() + f'/GradientBoostingRegressor_{name}.pkl', 'wb'))
    print(f"[INFO] {name} SAVED")

def train_eval(cam_mask_label, y_true):
    X_train, X_test, y_train, y_test = train_test_split(cam_mask_label, y_true, test_size=0.2, random_state=0)
    print(f"MAX VAL {X_train.max()}")
    scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    model = GradientBoostingRegressor(verbose=0)
    model.fit(X_train_scaled, y_train)


    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    return scaler, model, (X_test_scaled, y_pred, X_test)

scaler_resnest_0, model_resnest_0, pred_resnest_0  = train_eval(cam_mask_label_resnest_0, y_true_resnest_0)
scaler_resnest_1, model_resnest_1, pred_resnest_1  = train_eval(cam_mask_label_resnest_1, y_true_resnest_1)

scaler_effnet_0, model_effnet_0, pred_effnet_0 = train_eval(cam_mask_label_effnet_0, y_true_effnet_0)
scaler_effnet_1, model_effnet_1, pred_effnet_1 = train_eval(cam_mask_label_effnet_1, y_true_effnet_1)
scaler_effnet_2, model_effnet_2, pred_effnet_2 = train_eval(cam_mask_label_effnet_2, y_true_effnet_2)

save_scaler_model(scaler_resnest_0, model_resnest_0, "resnest0_2")
save_scaler_model(scaler_resnest_1, model_resnest_1, "resnest1_2")

save_scaler_model(scaler_effnet_0, model_effnet_0, "effnet0_2")
save_scaler_model(scaler_effnet_1, model_effnet_1, "effnet1_2")
save_scaler_model(scaler_effnet_2, model_effnet_2, "effnet2_2")


