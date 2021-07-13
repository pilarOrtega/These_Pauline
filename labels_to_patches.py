import os
import argparse
import logging
import pandas as pd
from data import get_patch_folders_in_project, get_patch_csv_from_patch_folder
from unidecode import unidecode


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
parser.add_argument("--cohort", type=str, default='Lymphopath',
                    help='name of the cohort')

args = parser.parse_args()


def main():
    data = pd.read_csv(args.data, sep=None, engine='python')
    proj_dir = args.projdir
    cohort = args.cohort

    for ptc_folder in get_patch_folders_in_project(proj_dir):
        try:
            patches_csv = get_patch_csv_from_patch_folder(ptc_folder)
            patches = pd.read_csv(patches_csv, sep=None, engine='python')
            slidename = os.path.basename(ptc_folder)
            if cohort in ['Lymphopath']:
                slidename_short = slidename.split('_')[2]
                slidename_short = slidename_short[:9]
            if cohort in ['Lysa']:
                slidename_short = slidename.split(' ')[3].split('-')[0]
                if slidename_short[0] == '0':
                    slidename_short = slidename_short[1:]
            type = os.path.dirname(ptc_folder)
            type = os.path.basename(type)
            type = type.replace(' ', '_')
            type = unidecode(type)
            size = len(patches)
            choices = {'DHL_BCL2': ('R', 'R', 'R', 'T'),
                       'DHL_BCL6': ('R', 'R', 'R', 'T'),
                       'DLBCL_sans_rearrangement': ('NR', 'NR', 'NR', 'T'),
                       'THL': ('R', 'R', 'R', 'T'),
                       'Burkitt': ('NA', 'NA', 'R', 'T'),
                       'MYC_rearrange_seul': ('NA', 'R', 'R', 'T'),
                       'MYC_rearrange_seul_ou_DHL_BCL6': ('NA', 'R', 'R', 'T'),
                       'Architecture_ganglion_normal': ('NA', 'NA', 'NA', 'N'),
                       'Architecture_ganglion_normale': ('NA', 'NA', 'NA', 'N'),
                       'Artefacts': ('NA', 'NA', 'NA', 'N'),
                       'Autres_tissus': ('NA', 'NA', 'NA', 'N')}
            t1, t2, t3, t5 = choices.get(type, ('NA', 'NA', 'NA', 'NA'))
            patches['Type'] = [type for n in range(size)]
            patches['Task_1'] = [t1 for n in range(size)]
            patches['Task_2'] = [t2 for n in range(size)]
            patches['Task_3'] = [t3 for n in range(size)]
            patches['Task_5'] = [t5 for n in range(size)]
            try:
                index = data[data['NUM_anapath'].str.contains(slidename_short)].index[0]
                patient_row = data.loc[index]
                if type in ['Architecture_ganglion_normal', 'Autres_tissus', 'Artefacts']:
                    patches['Task_4'] = ['NA' for n in range(size)]
                else:
                    patches['Task_4'] = [patient_row['GC_non_GC'] for n in range(size)]
            except IndexError as e:
                logger.warning(f'Slide {slidename_short} not found')
                # rewrite dataframe
            patches.to_csv(patches_csv, index=False)
        except PatchesNotFoundError as e:
            logger.warning(str(e))


if __name__ == "__main__":
    main()
