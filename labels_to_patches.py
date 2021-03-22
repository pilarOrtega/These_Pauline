import os
import argparse
import logging
import pandas as pd
from glob import glob


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


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('labels_on_projet_pauline.log')
c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.DEBUG)
c_format = logging.Formatter('%(name)s - [%(levelname)s] - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)
logger.addHandler(c_handler)
logger.addHandler(f_handler)

parser = argparse.ArgumentParser()

parser.add_argument("--data", type=str,
                    help="data file.")
parser.add_argument("--projdir", type=str,
                    help="pathaia dataset directory.")

args = parser.parse_args()


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


def main():
    data = pd.read_csv(args.data, sep=None, engine='python')
    data.set_index("NUM_anapath", inplace=True)
    proj_dir = args.projdir

    for slidename, ptc_folder in get_patch_folders_in_project(proj_dir):
        try:
            patches_csv = get_patch_csv_from_patch_folder(ptc_folder)
            patches = pd.read_csv(patches_csv, sep=None, engine='python')
            slidename_short = slidename.split('_')[2]
            slidename_short = slidename_short[:9]
            type = os.path.dirname(ptc_folder)
            type = os.path.basename(type)
            size = len(patches)
            choices = {'DHL_BCL2': ('R', 'R', 'R', 'T'),
                       'DHL_BCL6': ('R', 'R', 'R', 'T'),
                       'DLBCL_sans_rearrangement': ('NR', 'NR', 'NR', 'T'),
                       'THL': ('R', 'R', 'R', 'T'),
                       'Burkitt': ('NA', 'NA', 'R', 'T'),
                       'MYC_rearrange_seul': ('NA', 'R', 'R', 'T'),
                       'MYC_rearrange_seul_ou_DHL_BCL6': ('NA', 'R', 'R', 'T'),
                       'Architecture_ganglion_normal': ('NA', 'NA', 'NA', 'N'),
                       'Artefacts': ('NA', 'NA', 'NA', 'N'),
                       'Autres_tissus': ('NA', 'NA', 'NA', 'N')}
            t1, t2, t3, t5 = choices.get(type, ('NA', 'NA', 'NA'))
            patches['Type'] = [type for n in range(size)]
            patches['Task_1'] = [t1 for n in range(size)]
            patches['Task_2'] = [t2 for n in range(size)]
            patches['Task_3'] = [t3 for n in range(size)]
            patches['Task_5'] = [t5 for n in range(size)]
            try:
                patient_row = data.loc[slidename_short]
                if type in ['Architecture_ganglion_normal', 'Autres_tissus', 'Artefacts']:
                    patches['Task_4'] = ['N' for n in range(size)]
                else:
                    patches['Task_4'] = [patient_row['GC_non_GC'] for n in range(size)]
            except KeyError as e:
                logger.warning(str(e))
                # rewrite dataframe
            patches.to_csv(patches_csv)
        except PatchesNotFoundError as e:
            logger.warning(str(e))


if __name__ == "__main__":
    main()
