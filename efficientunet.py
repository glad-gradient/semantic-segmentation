from datetime import datetime
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed

from keras import backend as K
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras import layers

from segmentation_models import Unet
from segmentation_models import losses
from efficientnet.keras import preprocess_input

from utils import preprocess_data, make_image_generator


RANDOM_SEED = 10000
MODEL_NAME = 'efficientunet'

seed(RANDOM_SEED)

binary_focal_loss = losses.BinaryFocalLoss(alpha=10.0)
dice_loss = losses.DiceLoss(smooth=1)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def binary_focal_dice_loss(y_true, y_pred):
    return binary_focal_loss(y_true, y_pred) + dice_loss(y_true, y_pred)


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


if __name__ == '__main__':
    with open('config.json', 'r') as file:
        configs = json.load(file)

    model_configs = configs['UNET']

    balanced = model_configs['BALANCE_DATA']
    batch_size = model_configs['BATCH_SIZE']
    epochs = model_configs['EPOCHS']
    steps_per_epoch = model_configs['STEPS_PER_EPOCH']
    img_size = model_configs['IMG_SIZE']
    learning_rate = model_configs['LEARNING_RATE']
    backbone = model_configs['BACKBONE']
    path_checkpoint = model_configs['PATH_CHECKPOINT']
    model_to_load = model_configs['LOAD_MODEL']
    pretrained_encoder_weights = model_configs['PRETRAINED_ENCODER_WEIGHTS']

    data_dir = configs["DATA_DIRECTORY"]
    train_image_dir = os.path.join(data_dir, 'train_v2')

    corrupted_images = configs['EXCLUDE_IMAGES']
    train_df, valid_df = preprocess_data(
        os.path.join(data_dir, 'train_ship_segmentations_v2.csv.zip'),
        corrupted_images,
        n_train=0.85,
        balanced=balanced
    )

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

    train_generator = create_aug_generator(
        make_image_generator(in_df=train_df, batch_size=batch_size, img_size=img_size, train_image_dir=train_image_dir),
        image_generator,
        label_generator
    )
    valid_generator = create_aug_generator(
        make_image_generator(in_df=valid_df, batch_size=batch_size, img_size=img_size, train_image_dir=train_image_dir),
        image_generator,
        label_generator
    )

    K.set_image_data_format('channels_last')

    if pretrained_encoder_weights:
        model = Unet(
            backbone_name=backbone,
            input_shape=(None, None, 3),
            classes=1,
            activation='sigmoid',
            encoder_weights=pretrained_encoder_weights,
            encoder_freeze=True,
            decoder_block_type='transpose',
            decoder_filters=(256, 128, 64, 32, 16),
            decoder_use_batchnorm=True
        )
    else:
        MODEL_NAME = model_to_load

        model = load_model(f'models/{model_to_load}.hdf5', compile=False)

        model_split = 'block2a_expand_conv'
        layer_idx = [i for i in range(len(model.layers)) if model.layers[i].name == model_split][0]

        for i, layer in enumerate(model.layers):
            if i < layer_idx or isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss=binary_focal_dice_loss,
        metrics=[dice_coef],
    )

    no_train_steps = min(steps_per_epoch, len(train_df) // batch_size)
    no_valid_steps = min(steps_per_epoch, len(valid_df) // batch_size)

    early_stop_callback = EarlyStopping(monitor='val_dice_coef',
                                        patience=15,
                                        mode='max',
                                        verbose=1)

    model_id = int(datetime.utcnow().timestamp())

    checkpoint_callback = ModelCheckpoint(filepath=f'{path_checkpoint}/{MODEL_NAME}_{img_size}_{model_id}.hdf5',
                                          monitor='val_dice_coef',
                                          mode='max',
                                          verbose=1,
                                          save_best_only=True)

    history = model.fit_generator(
        train_generator,
        epochs=epochs,
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
    plt.savefig(f'learning_curves/{MODEL_NAME}_{img_size}_{model_id}.png')
