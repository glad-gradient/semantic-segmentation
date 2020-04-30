import os

import numpy as np
from pandas import read_csv, merge, concat
from skimage.io import imread
from cv2 import resize


def preprocess_data(file_path, exclude_list, n_train=0.8, balanced=False):
    masks = read_csv(file_path)

    masks = masks[~masks['ImageId'].isin(exclude_list)]

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

    train_df = merge(masks, train_ids)
    valid_df = merge(masks, valid_ids)

    return train_df, valid_df


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


def make_image_generator(in_df, batch_size, img_size, train_image_dir, inference_mode=False, preprocessing_function=None):
    all_batches = list(in_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(train_image_dir, c_img_id)
            _img = imread(rgb_path)
            _img = resize(_img, (img_size, img_size))

            _mask = masks_as_image(c_masks['EncodedPixels'].values)
            _mask = resize(_mask, (img_size, img_size))
            _mask = np.expand_dims(_mask, -1)

            _img = _img.astype(np.float32)
            _mask = _mask.astype(np.uint8)

            if preprocessing_function:
                _img = preprocessing_function(_img)

            out_rgb += [_img]
            out_mask += [_mask]
            if len(out_rgb) >= batch_size:
                if inference_mode:
                    yield np.stack(out_rgb, 0), np.stack(out_mask, 0), c_img_id
                else:
                    yield np.stack(out_rgb, 0), np.stack(out_mask, 0)
                out_rgb, out_mask = [], []

