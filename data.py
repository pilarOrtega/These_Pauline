"""A module to handle data generation for deep neural networks.
It uses the keras Sequence object to enable parallel computing of batches.
"""
import numpy as np
import pandas as pd
import keras
import os
import openslide
import warnings
import tensorflow as tf


class Error(Exception):
    """
    Base of custom errors.

    **********************
    """

    pass


class LevelNotFoundError(Error):
    """
    Raise when trying to access unknown level.

    *********************************************
    """

    pass


class EmptyProjectError(Error):
    """
    Raise when trying to access unknown level.

    *********************************************
    """

    pass


class SlideNotFoundError(Error):
    """
    Raise when trying to access unknown level.

    *********************************************
    """

    pass


class PatchesNotFoundError(Error):
    """
    Raise when trying to access unknown level.

    *********************************************
    """

    pass


class UnknownColumnError(Error):
    """
    Raise when trying to access unknown level.

    *********************************************
    """

    pass


def get_patch_csv_from_patch_folder(patch_folder):
    if os.path.isdir(patch_folder):
        patch_file = os.path.join(patch_folder, "patches.csv")
        if os.path.exists(patch_file):
            return patch_file
        raise PatchesNotFoundError(
            "Could not find extracted patches for the slide: {}".format(patch_folder)
        )
    raise SlideNotFoundError(
        "Could not find a patch folder at: {}!!!".format(patch_folder)
    )


def get_patch_folders_in_project(project_folder):
    if not os.path.isdir(project_folder):
        raise EmptyProjectError(
            "Did not find any project at: {}".format(project_folder)
        )
    for name in os.listdir(project_folder):
        if "annotation" not in name:
            patch_folder = os.path.join(project_folder, name)
            if os.path.isdir(patch_folder):
                yield name, patch_folder


def get_slide_file(slide_folder, slide_name, patch_folder=''):
    if not os.path.isdir(slide_folder):
        raise SlideNotFoundError(
            "Could not find a slide folder at: {}!!!".format(slide_folder)
        )
    if patch_folder != '':
        category = os.path.basename(os.path.dirname(patch_folder))
        slide_folder = os.path.join(slide_folder, category)
    for name in os.listdir(slide_folder):
        if name.endswith(".mrxs") and not name.startswith("."):
            base, _ = os.path.splitext(name)
            if slide_name == base:
                return os.path.join(slide_folder, name)
    raise SlideNotFoundError(
        "Could not find an mrxs slide file for: {} in {}!!!".format(slide_name,
                                                                    slide_folder)
    )


def handle_patch_file(patch_file, level, column):
    df = pd.read_csv(patch_file)
    level_df = df[df["level"] == level]
    if column not in level_df:
        raise UnknownColumnError(
            "Column {} does not exists in {}!!!".format(column, patch_file)
        )
    for _, row in level_df.iterrows():
        yield row["x"], row["y"], row[column]


class PathaiaHandler(object):

    def __init__(self, project_folder, slide_folder):
        self.slide_folder = slide_folder
        self.project_folder = project_folder

    def list_patches(self, level, dim, label):
        patch_list = []
        labels = []
        for name, patch_folder in get_patch_folders_in_project(self.project_folder):
            try:
                slide_path = get_slide_file(self.slide_folder, name, patch_folder)
                patch_file = get_patch_csv_from_patch_folder(patch_folder)
                slide_name = name.split('_')[2]
                slide_name = slide_name[:9]
                # read patch file and get the right level
                for x, y, lab in handle_patch_file(patch_file, level, label):
                    patch_list.append(
                        {"slide_path": slide_path, "slide": slide_name,
                         "x": x, "y": y, "level": level, "dimensions": dim}
                    )
                    labels.append(lab)
            except (PatchesNotFoundError, UnknownColumnError, SlideNotFoundError) as e:
                warnings.warn(str(e))
        return patch_list, labels


def slide_query(patch):
    slide = openslide.OpenSlide(patch["slide_path"])
    pil_img = slide.read_region((patch["x"], patch["y"]),
                                patch["level"], patch["dimensions"])
    return np.array(pil_img)[:, :, 0:3]


class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, labels, preproc,
                 batch_size=32, dim=(256, 256),
                 n_channels=3,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.preproc = preproc
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Put image data into a batch array
            X[i, ] = slide_query(self.list_IDs[ID])

            # Store class
            y[i] = self.labels[ID]

        return self.preproc(X), keras.utils.to_categorical(y, num_classes=self.n_classes)


def generator_fn(patch_list, label_list, preproc):

    def generator():
        for patch, y in zip(patch_list, label_list):
            x = slide_query(patch)
            yield preproc(x), y
    return generator


def get_tf_dataset(patch_list, label_list, preproc, batch_size, patch_size):
    gen = generator_fn(patch_list, label_list, preproc)
    dataset = tf.data.Dataset.from_generator(
        generator=gen,
        output_types=(np.float32, np.int32),
        output_shapes=((patch_size, patch_size, 3), label_list[0].shape)
    )
    dataset = dataset.batch(batch_size, drop_remainder=True)
    # prefetch
    # <=> while fitting batch b, prepare b+1 in parallel
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset