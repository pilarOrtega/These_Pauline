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
        tag = str(tags[t])
        patch = ptcs[t]
        if tag in unique_tags and tag not in ["NA", 'nan']:
            labels.append(tag)
            patches.append(patch)
    label_dict = get_label_dict(labels)
    labels = [label_dict[t] for t in labels]
    labels = np.array(labels)
    patches = np.array(patches)
    return patches, labels, label_dict
