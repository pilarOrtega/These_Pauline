import os
from glob import glob
import numpy as np


class Error(Exception):
    """
    Base of custom errors.

    **********************
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


# Transforms labels to numerical value
def get_label_dict(labels):
    outcomes = np.unique(labels)
    return {v: k for k, v in enumerate(outcomes)}


# Returns array of patches and array of numerical labels
def get_whole_dataset(ptcs, tags):
    labels = []
    patches = []
    unique_tags = np.unique(tags)
    for t in range(len(tags)):
        tag = tags[t]
        patch = ptcs[t]
        if tag in unique_tags and tag != "NA":
            labels.append(tag)
            patches.append(patch)
    label_dict = get_label_dict(labels)
    labels = [label_dict[t] for t in labels]
    labels = np.array(labels)
    patches = np.array(patches)
    return patches, labels, label_dict
