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
from glob import glob
from nptyping import NDArray
from typing import List, Optional, Any
from numbers import Number
from staintools.miscellaneous.get_concentrations import get_concentrations
import albumentations as a
from openslide import OpenSlide


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
    for class_folder in glob(os.path.join(project_folder, '*')):
        for name in os.listdir(class_folder):
            patch_folder = os.path.join(class_folder, name)
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
    if column == 'Unlabeled':
        for _, row in level_df.iterrows():
            yield row["x"], row["y"], column, row["dx"], row["dy"]
    elif column not in level_df:
        raise UnknownColumnError(
            "Column {} does not exists in {}!!!".format(column, patch_file)
        )
    for _, row in level_df.iterrows():
        yield row["x"], row["y"], row[column], row["dx"], row["dy"]


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
                # read patch file and get the right level
                for x, y, lab, dx, dy in handle_patch_file(patch_file, level, label):
                    patch_list.append(
                        {"slide_path": slide_path, "slide": slide_name,
                         "x": x, "y": y, "level": level, "dimensions": dim, "dx": dx, "dy": dy}
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


def random_modif(patch, alpha=0.2, beta=0.2):
    x = np.random.uniform(low=-1, size=2)
    patch["x"] = int(patch["x"] + 224*alpha*x[0])
    patch["y"] = int(patch["y"] + 224*beta*x[1])
    return patch


def ifnone(a: Any, b: Any) -> Any:
    "`b` if `a` is None else `a`"
    return b if a is None else a


class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, labels, preproc,
                 batch_size=32, dim=(256, 256),
                 n_channels=3, balanced=True,
                 num_samples=None, replacement=True,
                 data_augmentation=False,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.classes = np.unique(labels)
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.preproc = preproc
        self.balanced = balanced
        self.num_samples = num_samples
        self.replacement = replacement
        self.data_augmentation = data_augmentation
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.balanced:
            self.tree = {}
            patch_slides = [x['slide'] for x in self.list_IDs]
            patch_slides = np.asarray(patch_slides)
            for cl in np.arange(len(self.classes)):
                self.tree[cl] = {}
                slides = [self.list_IDs[i]['slide'] for i in range(len(self.labels)) if self.labels[i] == cl]
                for slide in np.unique(slides):
                    self.tree[cl][slide] = (np.argwhere(patch_slides == slide).squeeze(1).tolist())

            n_slides = [len(self.tree[c]) for c in self.tree.keys()]
            self.num_samples = 0
            for c in self.tree.keys():
                n_patches = [len(self.tree[c][s]) for s in self.tree[c].keys()]
                for i in range(min(n_slides)):
                    self.num_samples += min(500, n_patches[i])
            self.indexes = self.get_idxs()
        else:
            self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def get_idxs(self):
        idxs = []
        for _ in range(self.num_samples):
            x = np.random.uniform(size=3)
            classes = list(self.tree.keys())
            cl = classes[int(x[0]*len(classes))]
            cl_slides = self.tree[cl]
            slides = list(cl_slides.keys())
            slide = slides[int(x[1]*len(slides))]
            slide_patches = cl_slides[slide]
            idx = int(x[2]*len(slide_patches))
            if self.replacement:
                patch = slide_patches[idx]
            else:
                patch = slide_patches.pop(idx)
                if len(slide_patches) == 0:
                    cl_slides.pop(slide)
                    if len(cl_slides) == 0:
                        self.tree.pop(cl)
            idxs.append(patch)
        return idxs

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Put image data into a batch array
            patch = self.list_IDs[ID]
            if self.data_augmentation:
                patch = random_modif(patch)
            X[i, ] = slide_query(patch)

            # Store class
            y[i] = self.labels[ID]

        return self.preproc(X), keras.utils.to_categorical(y, num_classes=self.n_classes)


def generator_generator(patches, labels):
    def generator():
        slide_list = [p["slide"] for p in patches]
        slide_set = np.unique(slide_list)
        slides = {s: OpenSlide(s) for s in slide_set}
        for patch, label in zip(patches, labels):
            slide = slides[patch["slide"]]
            pil_img = slide.read_region(
                (patch["x"], patch["y"]),
                patch["level"],
                patch["dimensions"]
            )
            x = np.array(pil_img)[:, :, 0:3]
            yield x, label
    return generator


def create_dataset(patches, labels, PATCH_SIZE=224, BATCH=16, PREFETCH=None):
    gen = generator_generator(patches, labels)
    dataset = tf.data.Dataset.from_generator(
        generator=gen,
        output_types=(np.float32, np.int32),
        output_shapes=((PATCH_SIZE, PATCH_SIZE, 3), labels[0].shape)
    )
    if PREFETCH is not None:
        return dataset.batch(BATCH).prefetch(PREFETCH)
    else:
        return dataset.batch(BATCH)

# Cell
class StainAugmentor(a.ImageOnlyTransform):
    def __init__(
        self,
        alpha_range: float = 0.3,
        beta_range: float = 0.3,
        alpha_stain_range: float = 0.2,
        beta_stain_range: float = 0.1,
        he_ratio: float = 0.2,
        always_apply: bool = True,
        p: float = 1,
    ):
        super(StainAugmentor, self).__init__(always_apply, p)
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.alpha_stain_range = alpha_stain_range
        self.beta_stain_range = beta_stain_range
        self.he_ratio = he_ratio
        self.stain_matrix = np.array(
            [[0.56371366, 0.77129725, 0.29551221], [0.1378605, 0.82185632, 0.55276276]]
        )

    def get_params(self):
        return {
            "alpha": np.random.uniform(
                1 - self.alpha_range, 1 + self.alpha_range, size=2
            ),
            "beta": np.random.uniform(-self.beta_range, self.beta_range, size=2),
            "alpha_stain": np.stack(
                (
                    np.random.uniform(
                        1 - self.alpha_stain_range * self.he_ratio,
                        1 + self.alpha_stain_range * self.he_ratio,
                        size=3,
                    ),
                    np.random.uniform(
                        1 - self.alpha_stain_range,
                        1 + self.alpha_stain_range,
                        size=3,
                    ),
                ),
            ),
            "beta_stain": np.stack(
                (
                    np.random.uniform(
                        -self.beta_stain_range * self.he_ratio,
                        self.beta_stain_range * self.he_ratio,
                        size=3,
                    ),
                    np.random.uniform(
                        -self.beta_stain_range, self.beta_stain_range, size=3
                    ),
                ),
            ),
        }

    def initialize(self, alpha, beta, shape=2):
        alpha = ifnone(alpha, np.ones(shape))
        beta = ifnone(beta, np.zeros(shape))
        return alpha, beta

    def apply(
        self,
        image: NDArray[(Any, Any, 3), Number],
        alpha: Optional[NDArray[(2,), float]] = None,
        beta: Optional[NDArray[(2,), float]] = None,
        alpha_stain: Optional[NDArray[(2, 3), float]] = None,
        beta_stain: Optional[NDArray[(2, 3), float]] = None,
        **params
    ) -> NDArray[(Any, Any, 3), Number]:
        alpha, beta = self.initialize(alpha, beta, shape=2)
        alpha_stain, beta_stain = self.initialize(alpha_stain, beta_stain, shape=(2, 3))
        if not image.dtype == np.uint8:
            image = (image * 255).astype(np.uint8)
        # stain_matrix = VahadaneStainExtractor.get_stain_matrix(image)
        HE = get_concentrations(image, self.stain_matrix)
        # HE = convert_RGB_to_OD(image).reshape((-1, 3)) @ np.linalg.pinv(self.stain_matrix)
        stain_matrix = self.stain_matrix * alpha_stain + beta_stain
        stain_matrix = np.clip(stain_matrix, 0, 1)
        HE = np.where(HE > 0.2, HE * alpha[None] + beta[None], HE)
        out = np.exp(-np.dot(HE, stain_matrix)).reshape(image.shape)
        out = np.clip(out, 0, 1)
        return out.astype(np.float32)

    def get_transform_init_args_names(self) -> List:
        return ("alpha_range", "beta_range", "alpha_stain_range", "beta_stain_range", "he_ratio")


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
