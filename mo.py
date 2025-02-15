import os
import json

import cv2
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.losses import binary_crossentropy
from keras.callbacks import Callback, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
import time
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

name = 'SDD_{}'.format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(name), write_grads=True, write_graph=True, histogram_freq=1)
# tensorboard = TensorBoard(log_dir='logs/{}'.format(name), write_grads=True, write_graph=True)

train_df = pd.read_csv('data/train.csv')
train_df['ImageId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
train_df['ClassId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[1])
train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()

# print(train_df.shape)
# train_df.head()

mask_count_df = train_df.groupby('ImageId').agg(np.sum).reset_index()
mask_count_df.sort_values('hasMask', ascending=False, inplace=True)
# print(mask_count_df.shape)
# mask_count_df.head()

def rle2mask(mask_rle, shape=(1600, 256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def build_masks(rles, input_shape):
    depth = len(rles)
    height, width = input_shape
    masks = np.zeros((height, width, depth))

    for i, rle in enumerate(rles):
        if type(rle) is str:
            masks[:, :, i] = rle2mask(rle, (width, height))

    return masks

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)









# sample_filename = 'db4867ee8.jpg'
# sample_image_df = train_df[train_df['ImageId'] == sample_filename]
# sample_path = 'data/train_images/{}'.format(sample_filename)
# sample_img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
# sample_rles = sample_image_df['EncodedPixels'].values
# sample_masks = build_masks(sample_rles, input_shape=(256, 1600))
# sample_img = np.expand_dims(sample_img, axis=-1)
# print(sample_img.shape)
# print(sample_masks.shape)
#
# fig, axs = plt.subplots(5, figsize=(12, 12))
# axs[0].imshow(sample_img[:,:,0], cmap='gray')
# axs[0].axis('off')
#
# for i in range(4):
#     axs[i+1].imshow(sample_masks[:, :, i])
#     axs[i+1].axis('off')


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, df, target_df=None, mode='fit',
                 base_path='data/train_images',
                 batch_size=2, dim=(256, 1600), n_channels=1,
                 n_classes=4, random_state=2019, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.mode = mode
        self.base_path = base_path
        self.target_df = target_df
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.random_state = random_state
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_batch = [self.list_IDs[k] for k in indexes]

        X = self.__generate_X(list_IDs_batch)

        if self.mode == 'fit':
            y = self.__generate_y(list_IDs_batch)
            return X, y

        elif self.mode == 'predict':
            return X

        else:
            raise AttributeError('The mode parameter should be set to "fit" or "predict".')

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)

    def __generate_X(self, list_IDs_batch):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['ImageId'].iloc[ID]
            img_path = '{}/{}'.format(self.base_path, im_name)
            img = self.__load_grayscale(img_path)

            # Store samples
            X[i,] = img

        return X

    def __generate_y(self, list_IDs_batch):
        y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)

        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['ImageId'].iloc[ID]
            image_df = self.target_df[self.target_df['ImageId'] == im_name]

            rles = image_df['EncodedPixels'].values
            masks = build_masks(rles, input_shape=self.dim)

            y[i,] = masks

        return y

    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.
        img = np.expand_dims(img, axis=-1)

        return img

    # def __load_rgb(self, img_path):
    #     img = cv2.imread(img_path)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img = img.astype(np.float32) / 255.
    #
    #     return img


BATCH_SIZE = 8

# train_idx, val_idx = train_test_split(
#     mask_count_df.index, random_state=777, test_size=0.15
# )

X_train, test_idx = train_test_split(mask_count_df.index, test_size=0.2, random_state=1)
train_idx, val_idx = train_test_split(X_train, test_size=0.125, random_state=1)


train_generator = DataGenerator(
    train_idx,
    df=mask_count_df,
    target_df=train_df,
    batch_size=BATCH_SIZE,
    n_classes=4
)

# val_generator = DataGenerator(
#     val_idx,
#     df=mask_count_df,
#     target_df=train_df,
#     batch_size=BATCH_SIZE,
#     n_classes=4
# )
#
X_val = np.empty((len(val_idx), 256, 1600, 1))

for i, ID in enumerate(val_idx):
    im_name = mask_count_df['ImageId'].iloc[ID]
    img_path = 'data/train_images/{}'.format(im_name)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.
    img = np.expand_dims(img, axis=-1)

    X_val[i,] = img

y_val = np.empty(((len(val_idx), 256, 1600, 4)), dtype=int)
for i, ID in enumerate(val_idx):
    im_name = mask_count_df['ImageId'].iloc[ID]
    image_df = train_df[train_df['ImageId'] == im_name]

    rles = image_df['EncodedPixels'].values
    masks = build_masks(rles, input_shape=(256,1600))

    y_val[i,] = masks



def build_model(input_shape):
    inputs = Input(input_shape)

    c1 = Conv2D(8, (3, 3), activation='elu', padding='same')(inputs)
    c1 = Conv2D(8, (3, 3), activation='elu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(16, (3, 3), activation='elu', padding='same')(p1)
    c2 = Conv2D(16, (3, 3), activation='elu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(32, (3, 3), activation='elu', padding='same')(p2)
    c3 = Conv2D(32, (3, 3), activation='elu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(64, (3, 3), activation='elu', padding='same')(p3)
    c4 = Conv2D(64, (3, 3), activation='elu', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(64, (3, 3), activation='elu', padding='same')(p4)
    c5 = Conv2D(64, (3, 3), activation='elu', padding='same')(c5)
    p5 = MaxPooling2D(pool_size=(2, 2))(c5)

    c55 = Conv2D(128, (3, 3), activation='elu', padding='same')(p5)
    c55 = Conv2D(128, (3, 3), activation='elu', padding='same')(c55)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c55)
    u6 = concatenate([u6, c5])
    c6 = Conv2D(64, (3, 3), activation='elu', padding='same')(u6)
    c6 = Conv2D(64, (3, 3), activation='elu', padding='same')(c6)

    u71 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u71 = concatenate([u71, c4])
    c71 = Conv2D(32, (3, 3), activation='elu', padding='same')(u71)
    c61 = Conv2D(32, (3, 3), activation='elu', padding='same')(c71)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c61)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='elu', padding='same')(u7)
    c7 = Conv2D(32, (3, 3), activation='elu', padding='same')(c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='elu', padding='same')(u8)
    c8 = Conv2D(16, (3, 3), activation='elu', padding='same')(c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='elu', padding='same')(u9)
    c9 = Conv2D(8, (3, 3), activation='elu', padding='same')(c9)

    outputs = Conv2D(4, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    ADam = keras.optimizers.adam(lr=0.0008)
    model.compile(optimizer=ADam, loss=bce_dice_loss, metrics=[dice_coef])

    return model


model = build_model((256, 1600, 1))
model.summary()

history = model.fit_generator(train_generator, validation_data=(X_val, y_val), use_multiprocessing=False,
                              workers=1, epochs=1, callbacks=[tensorboard])

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")


with open('history.json', 'w') as f:
    json.dump(history.history, f)

history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['dice_coef', 'val_dice_coef']].plot()

model.load_weights('model.h5')


def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def build_rles(masks):
    width, height, depth = masks.shape

    rles = [mask2rle(masks[:, :, i])
            for i in range(depth)]

    return rles

X_test = np.empty((len(test_idx), 256, 1600, 1))

for i, ID in enumerate(test_idx):
    im_name = mask_count_df['ImageId'].iloc[ID]
    img_path = 'data/train_images/{}'.format(im_name)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.
    img = np.expand_dims(img, axis=-1)

    X_test[i,] = img

y_test = np.empty(((len(test_idx), 256, 1600, 4)), dtype=int)
for i, ID in enumerate(test_idx):
    im_name = mask_count_df['ImageId'].iloc[ID]
    image_df = train_df[train_df['ImageId'] == im_name]

    rles = image_df['EncodedPixels'].values
    masks = build_masks(rles, input_shape=(256,1600))

    y_test[i,] = masks




sub_df = pd.read_csv('data/sample_submission.csv')
sub_df['ImageId'] = sub_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
test_imgs = pd.DataFrame(sub_df['ImageId'].unique(), columns=['ImageId'])

test_df = []

for i in range(0, test_imgs.shape[0], 500):
    batch_idx = list(
        range(i, min(test_imgs.shape[0], i + 500))
    )

    test_generator = DataGenerator(
        batch_idx,
        df=test_imgs,
        shuffle=False,
        mode='predict',
        base_path='data/test_images',
        target_df=sub_df,
        batch_size=1,
        n_classes=4
    )

    batch_pred_masks = model.predict_generator(
        test_generator,
        workers=1,
        verbose=1,
        use_multiprocessing=False
    )

    for j, b in tqdm(enumerate(batch_idx)):
        filename = test_imgs['ImageId'].iloc[b]
        image_df = sub_df[sub_df['ImageId'] == filename].copy()

        pred_masks = batch_pred_masks[j,].round().astype(int)
        pred_rles = build_rles(pred_masks)

        image_df['EncodedPixels'] = pred_rles
        test_df.append(image_df)

test_df = pd.concat(test_df)
test_df.drop(columns='ImageId', inplace=True)
test_df.to_csv('submission.csv', index=False)