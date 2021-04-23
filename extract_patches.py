## Extract patches with PATHAIA
## Code from patchify_goya

import argparse
import logging
from pathaia.patches import HierarchicalPatchifier, filters
from pathaia.patches.functional_api import *
import warnings
import os
import glob


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


parser = argparse.ArgumentParser()

parser.add_argument("--slides", type=str,
                    help="WSI dataset directory.")
parser.add_argument("--top", type=int,
                    help="top level of patch extraction.")
parser.add_argument("--low", type=int, default=0,
                    help='low level for patch extraction')
parser.add_argument("--psize", type=int,
                    help="size of patches.")
parser.add_argument("--outdir", type=str,
                    help="path to output patch files.")
parser.add_argument("--extension", type=str,
                    help="extension of WSI")
parser.add_argument("--verbose", type=int, default=2)
parser.add_argument("--dir_filter", type=str, default='*')

args = parser.parse_args()


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler("patchify_lymphopath.log")
c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.DEBUG)
c_format = logging.Formatter('%(name)s - [%(levelname)s] - %(message)s')
f_format = logging.Formatter(
    '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)
logger.addHandler(c_handler)
logger.addHandler(f_handler)


def has_tissue(img):
    mask = filters.get_tissue(img, blacktol=0, whitetol=230, method="rgb")
    return (mask.sum() > 0.8 * mask.shape[0] * mask.shape[1])


def main():
    inputdir = args.slides
    top = args.top
    low = args.low
    psize = args.psize
    outdir = args.outdir
    ext = args.extension
    verbose = args.verbose
    dir_filter = args.dir_filter

    logger.info("Processing slide folder {}".format(inputdir))
    logger.info("Processing from level {} to level 0".format(top))
    logger.info("Getting patches of size {}".format(psize))
    logger.info("Saving results in {}".format(outdir))

    filters = {k: [has_tissue] for k in range(top + 1)}
    silent = [k for k in range(top + 1)]

    try:
        os.mkdir(outdir)
        print("Directory", outdir, "created")
    except FileExistsError:
        print("Directory", outdir, "already exists")

    for folder in glob.glob(os.path.join(inputdir, dir_filter)):
        exit_folder = os.path.join(outdir, os.path.basename(folder))
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            patchify_folder_hierarchically(folder, exit_folder, top, low, psize, {
                                           'x': psize, 'y': psize}, filters=filters, silent=silent, extensions=(ext, ), verbose=verbose)
        for w in ws:
            logger.warning(w.message)


if __name__ == "__main__":
    main()
