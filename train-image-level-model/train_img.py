import efficientnet.tfkeras as efn
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from keras.callbacks import Callback
import tensorflow.keras.backend as K
from keras_applications import resnext
import keras
import tensorflow_addons as tfa
from vit_keras import vit, utils
from config import CFG
import os
from sklearn.preprocessing import MultiLabelBinarizer
from kaggle_datasets import KaggleDatasets
def auto_select_accelerator():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except ValueError:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    
    return strategy


def build_decoder(with_labels=True, target_size=(256, 256), ext='jpg'):
    def decode(path):
        if CFG.color_mode == "rgb":

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

        if CFG.color_mode == "ggg":
            g = tf.io.read_file(path + "_green.png")
            img = tf.image.decode_png(g, channels=3)
            img = tf.cast(img, tf.float32) / 255.0
            #if only green
            img = tf.image.resize(img, target_size)
        
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


def build_dataset(paths, labels=None, bsize=128, cache=True,
                  decode_fn=None, augment_fn=None,
                  augment=True, repeat=True, shuffle=1024, 
                  cache_dir=""):
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)
    
    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)
    
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

strategy = auto_select_accelerator()
BATCH_SIZE = strategy.num_replicas_in_sync * 16 #############WAS 16
GCS_DS_PATH = KaggleDatasets().get_gcs_path("hpa-768768")
GCS_DS_PATH_EXT_DATA = KaggleDatasets().get_gcs_path("hpa-public-768-excl-0-16")

load_dir = f"/kaggle/input/hpa-768768/"
df = pd.read_csv(os.getcwd() + 'train_data/df_green.csv')
df["ID"] = df["ID"].str.replace('_green', '')
label_cols = df.columns[2:21]
paths = GCS_DS_PATH + '/' + df['ID']
labels = df[label_cols].values

df_ext = pd.read_csv('train_data/hpa_public_excl_0_16_768.csv', index_col=0)
df_ext = df_ext.drop(['Cellline'], axis=1)
df_ext["Labels_list"] = df_ext["Label"].str.split("|").apply(lambda x: [int(i) for i in x])



mlb = MultiLabelBinarizer(classes=[n for n in range(19)])
y = df_ext["Labels_list"]

df_ohe = pd.DataFrame(mlb.fit_transform(y),columns=mlb.classes_)
df_ohe_np = df_ohe.to_numpy()

df_ext_ohe = pd.concat([df_ext, df_ohe], axis=1)
df_ext_ohe = df_ext_ohe.drop(['Labels_list'], axis=1)
df_ext_ohe.columns = df_ext_ohe.columns.astype(str)

label_cols_ext = df_ext_ohe.columns[2:21]
paths_ext = GCS_DS_PATH_EXT_DATA + '/hpa_public_excl_0_16_768/small/' + df_ext_ohe['ID']
labels_ext = df_ext_ohe[label_cols_ext].values

labels_all = np.append(labels, labels_ext, axis=0)
paths_all = paths.append(paths_ext, ignore_index=True)
#sanity check
name = paths_all[22000].split("/")[-1].split(".")[0]
label_real = df_ext_ohe.loc[df_ext_ohe["ID"] == name].Label
label_set = np.where(labels_all[22000] == 1)

assert  int(label_real) == int(label_set[0])
(
    train_paths, valid_paths, 
    train_labels, valid_labels
) = train_test_split(paths_all, labels_all, test_size=0.1, random_state=42)


decoder = build_decoder(with_labels=True, target_size=(600, 600))
test_decoder = build_decoder(with_labels=False, target_size=(600, 600))

train_dataset = build_dataset(
    train_paths, train_labels, bsize=BATCH_SIZE, decode_fn=decoder
)

valid_dataset = build_dataset(
    valid_paths, valid_labels, bsize=BATCH_SIZE, decode_fn=decoder,
    repeat=False, shuffle=False, augment=False
)

try:
    n_labels = train_labels.shape[1]
except:
    n_labels = 1

  
with strategy.scope():
    if CFG.model == "effb7":
        model = tf.keras.Sequential([
            efn.EfficientNetB7(
                input_shape=(600, 600, 3),
                weights='imagenet',
                include_top=False),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(n_labels, activation='sigmoid')
        ])
    if CFG.model == "vit":
        model = vit.vit_b16(
                image_size=384,
                activation='sigmoid',
                pretrained=True,
                include_top=True,
                pretrained_top=False,
                classes=19
            )

    if CFG.model == "resnext":
            model = tf.keras.Sequential([
        resnext.ResNeXt101(include_top=False, weights='imagenet', input_shape=(600, 600, 3), backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(n_labels, activation='sigmoid')
    ])


    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(multi_label=True)])
        
steps_per_epoch = train_paths.shape[0] // BATCH_SIZE
checkpoint = tf.keras.callbacks.ModelCheckpoint(f'{CFG.color_mode}_{CFG.model}_BCE_EPOCH{epoch:02d}-VAL{val_loss:.4f}.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, min_lr=1e-6, mode='min', verbose=1)

class CallbackGetLR(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr_with_decay = self.model.optimizer._decayed_lr(tf.float32)
        print("Learning Rate = ", K.eval(lr_with_decay))
        
print_lr = CallbackGetLR()

history = model.fit(
    train_dataset, 
    epochs=100,
    verbose=1,
    callbacks=[checkpoint,lr_reducer, print_lr],
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_dataset)