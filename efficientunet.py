from datetime import datetime
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
from pandas import read_csv, merge, concat
from skimage.io import imread
from cv2 import resize

from keras import backend as K
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras import layers

from segmentation_models import Unet
from segmentation_models import losses
from efficientnet.keras import preprocess_input


BATCH_SIZE = 2
EPOCHS = 50
STEPS_PER_EPOCH = 100
RANDOM_SEED = 10000
RESIZE_IMG = 3
IMG_SIZE = 512
LEARNING_RATE = 0.000005
ALPHA = 10.0
ENCODER_FREEZE = True

BACKBONE = 'efficientnetb0'
PATH_CHECKPOINT = 'models'
MODEL_NAME = 'efficientunet'
BUILD_NEW_MODEL = False
MODEL = 'efficientunet_256_1580639531_384_1580712972'
PRETRAINED_ENCODER_WEIGHTS = None

seed(RANDOM_SEED)

binary_focal_loss = losses.BinaryFocalLoss(alpha=ALPHA)
dice_loss = losses.DiceLoss(smooth=1)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def binary_focal_dice_loss(y_true, y_pred):
    return binary_focal_loss(y_true, y_pred) + dice_loss(y_true, y_pred)


def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype=np.int16)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return all_masks


def make_image_generator(in_df, batch_size=BATCH_SIZE):
    all_batches = list(in_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(train_image_dir, c_img_id)
            _img = imread(rgb_path)
            _img = resize(_img, (IMG_SIZE, IMG_SIZE))

            _mask = masks_as_image(c_masks['EncodedPixels'].values)
            _mask = resize(_mask, (IMG_SIZE, IMG_SIZE))
            _mask = np.expand_dims(_mask, -1)

            _img = _img.astype(np.float32)
            _mask = _mask.astype(np.uint8)

            out_rgb += [_img]
            out_mask += [_mask]
            if len(out_rgb) >= batch_size:
                yield np.stack(out_rgb, 0), np.stack(out_mask, 0)
                out_rgb, out_mask = [], []


def create_aug_generator(in_gen, _image_generator, _label_generator, seed=RANDOM_SEED):
    np.random.seed(seed)
    for in_x, in_y in in_gen:
        _seed = np.random.choice(range(seed))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        _x = _image_generator.flow(in_x,
                                   batch_size=in_x.shape[0],
                                   seed=_seed,
                                   shuffle=True)
        _y = _label_generator.flow(in_y,
                                   batch_size=in_x.shape[0],
                                   seed=_seed,
                                   shuffle=True)

        yield next(_x), next(_y)


def preprocess_data(_train_image_dir, _exclude_list, n_train=0.8, balanced=False):
    masks = read_csv(os.path.join(data_dir, 'train_ship_segmentations_v2.csv.zip'))

    masks = masks[~masks['ImageId'].isin(_exclude_list)]

    masks['ships'] = masks['EncodedPixels'].map(lambda row: 1 if isinstance(row, str) else 0)
    unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()

    unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x > 0 else 0.0)

    n_train_split = int(n_train * len(unique_img_ids))

    train_ids = unique_img_ids.iloc[:n_train_split]
    valid_ids = unique_img_ids.iloc[n_train_split:]

    if balanced:
        ones = train_ids.loc[train_ids['has_ship'] == 1]
        zeros = train_ids.loc[train_ids['has_ship'] == 0]
        zeros = zeros.sample(len(ones))
        train_ids = concat([ones, zeros])

    masks.drop(['ships'], axis=1, inplace=True)

    _train_df = merge(masks, train_ids)
    _valid_df = merge(masks, valid_ids)

    return _train_df, _valid_df


if __name__ == '__main__':
    exclude_list = ['6384c3e78.jpg', '13703f040.jpg', '14715c06d.jpg', '33e0ff2d5.jpg',
                    '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg',
                    'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                    'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg',
                    '66049e4ea.jpg', '973ab3eb2.jpg']  # corrupted images

    with open('config.json', 'r') as file:
        configs = json.load(file)

    data_dir = configs["DATA_DIRECTORY"]
    train_image_dir = os.path.join(data_dir, 'train_v2')

    train_df, valid_df = preprocess_data(train_image_dir, exclude_list, n_train=0.85, balanced=True)

    data_generator_args = dict(
        rotation_range=15,
        height_shift_range=0.05,
        width_shift_range=0.05,
        shear_range=0.2,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        data_format="channels_last",
        preprocessing_function=preprocess_input
    )

    image_generator = ImageDataGenerator(**data_generator_args)
    if 'preprocessing_function' in data_generator_args:
        data_generator_args.pop('preprocessing_function')
    label_generator = ImageDataGenerator(**data_generator_args)

    train_generator = create_aug_generator(make_image_generator(train_df), image_generator, label_generator)
    valid_generator = create_aug_generator(make_image_generator(valid_df), image_generator, label_generator)

    K.set_image_data_format('channels_last')

    if PRETRAINED_ENCODER_WEIGHTS:
        model = Unet(
            backbone_name=BACKBONE,
            input_shape=(None, None, 3),
            classes=1,
            activation='sigmoid',
            encoder_weights=PRETRAINED_ENCODER_WEIGHTS,
            encoder_freeze=ENCODER_FREEZE,
            decoder_block_type='transpose',
            decoder_filters=(256, 128, 64, 32, 16),
            decoder_use_batchnorm=True
        )
    else:
        MODEL_NAME = MODEL

        model = load_model(f'models/{MODEL}.hdf5', compile=False)

        model_split = 'block2a_expand_conv'
        layer_idx = [i for i in range(len(model.layers)) if model.layers[i].name == model_split][0]

        for i, layer in enumerate(model.layers):
            if i < layer_idx or isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

    model.compile(
        optimizer=Adam(lr=LEARNING_RATE),
        loss=binary_focal_dice_loss,
        metrics=[dice_coef],
    )

    no_train_steps = min(STEPS_PER_EPOCH, len(train_df) // BATCH_SIZE)
    no_valid_steps = min(STEPS_PER_EPOCH, len(valid_df) // BATCH_SIZE)

    early_stop_callback = EarlyStopping(monitor='val_dice_coef',
                                        patience=15,
                                        mode='max',
                                        verbose=1)

    model_id = int(datetime.utcnow().timestamp())

    checkpoint_callback = ModelCheckpoint(filepath=f'{PATH_CHECKPOINT}/{MODEL_NAME}_{IMG_SIZE}_{model_id}.hdf5',
                                          monitor='val_dice_coef',
                                          mode='max',
                                          verbose=1,
                                          save_best_only=True)

    history = model.fit_generator(
        train_generator,
        epochs=EPOCHS,
        steps_per_epoch=no_train_steps,
        validation_data=valid_generator,
        validation_steps=no_valid_steps,
        callbacks=[early_stop_callback, checkpoint_callback]
    )

    plt.figure(figsize=(10, 8), dpi=100)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Time')
    plt.legend(['Train', 'Valid'])
    plt.savefig(f'learning_curves/{MODEL_NAME}_{IMG_SIZE}_{model_id}.png')
