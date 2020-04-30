import json
import matplotlib.pyplot as plt
import os

import numpy as np
from skimage.io import imread
from cv2 import resize

from keras import backend as K
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras import layers
from keras.models import Model

from efficientnet.keras import EfficientNetB0
from efficientnet.keras import preprocess_input

from utils import preprocess_data


RANDOM_SEED = 10000
MODEL_NAME = 'efficientnet_classifier'


def make_image_generator(in_df, _batch_size, _img_size, _train_image_dir):
    all_batches = list(in_df.groupby('ImageId'))
    out_rgb = []
    labels = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(_train_image_dir, c_img_id)
            _img = imread(rgb_path)
            _img = resize(_img, (_img_size, _img_size))
            _img = _img.astype(np.float32)

            out_rgb += [_img]
            labels.append(c_masks['has_ship'].iloc[0])
            if len(out_rgb) >= _batch_size:
                yield np.stack(out_rgb, 0), np.array(labels)
                out_rgb, labels = [], []


def create_aug_generator(in_gen, _image_generator, seed=RANDOM_SEED):
    np.random.seed(seed)
    for in_x, in_y in in_gen:
        _seed = np.random.choice(range(seed))
        sample = _image_generator.flow(in_x,
                                       in_y,
                                       batch_size=in_x.shape[0],
                                       seed=_seed,
                                       shuffle=True)
        yield next(sample)


if __name__ == '__main__':
    with open('config.json', 'r') as file:
        configs = json.load(file)

    model_configs = configs['CLASSIFIER']

    balanced = model_configs['BALANCE_DATA']
    batch_size = model_configs['BATCH_SIZE']
    epochs = model_configs['EPOCHS']
    steps_per_epoch = model_configs['STEPS_PER_EPOCH']
    img_size = model_configs['IMG_SIZE']
    learning_rate = model_configs['LEARNING_RATE']
    path_checkpoint = model_configs['PATH_CHECKPOINT']
    model_to_load = model_configs['LOAD_MODEL']
    freeze = model_configs['FREEZE']

    data_dir = configs["DATA_DIRECTORY"]
    train_image_dir = os.path.join(data_dir, 'train_v2')

    corrupted_images = configs['EXCLUDE_IMAGES']
    train_df, valid_df = preprocess_data(
        os.path.join(data_dir, 'train_ship_segmentations_v2.csv.zip'),
        corrupted_images,
        n_train=0.85,
        balanced=balanced)

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

    data_generator = ImageDataGenerator(**data_generator_args)

    train_generator = create_aug_generator(
        make_image_generator(train_df, _batch_size=batch_size, _img_size=img_size, _train_image_dir=train_image_dir),
        data_generator
    )
    valid_generator = create_aug_generator(
        make_image_generator(valid_df, _batch_size=batch_size, _img_size=img_size, _train_image_dir=train_image_dir),
        data_generator
    )

    K.set_image_data_format('channels_last')

    if model_to_load:
        model = load_model(f'{path_checkpoint}/{model_to_load}.hdf5')
        MODEL_NAME = model_to_load

        model_split = 'block2a_expand_conv'

        layer_idx = [i for i in range(len(model.layers)) if model.layers[i].name == model_split][0]

        for i, layer in enumerate(model.layers):
            if i < layer_idx or isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
    else:
        base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(None, None, 3), pooling='avg')

        x = base_model.output
        predictions = layers.Dense(1, activation='sigmoid')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        if freeze:
            for layer in base_model.layers:
                layer.trainable = False
        else:
            model_split = 'block2a_expand_conv'

            layer_idx = [i for i in range(len(model.layers)) if model.layers[i].name == model_split][0]

            for i, layer in enumerate(model.layers):
                if i < layer_idx or isinstance(layer, layers.BatchNormalization):
                    layer.trainable = False
                else:
                    layer.trainable = True

    no_train_steps = min(steps_per_epoch, len(train_df) // batch_size)
    no_valid_steps = min(steps_per_epoch, len(valid_df) // batch_size)

    callbacks = []

    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    if model_to_load:
        FULL_MODEL_NAME = MODEL_NAME + '_retrained'
    else:
        FULL_MODEL_NAME = f'{MODEL_NAME}'

    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        patience=20,
                                        mode='min',
                                        verbose=1)

    checkpoint_callback = ModelCheckpoint(filepath=f'{path_checkpoint}/{FULL_MODEL_NAME}.hdf5',
                                          monitor='val_loss',
                                          mode='min',
                                          verbose=1,
                                          save_best_only=True)
    callbacks.append(early_stop_callback)
    callbacks.append(checkpoint_callback)

    history = model.fit_generator(
        train_generator,
        epochs=epochs,
        steps_per_epoch=no_train_steps,
        validation_data=valid_generator,
        validation_steps=no_valid_steps,
        callbacks=callbacks
    )

    plt.figure(figsize=(10, 8), dpi=100)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Time')
    plt.legend(['Train', 'Valid'])
    plt.savefig(f'learning_curves/{FULL_MODEL_NAME}.png')
