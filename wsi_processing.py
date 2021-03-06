import data
import os
import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.applications import xception
from auxiliary_functions import get_whole_dataset
from tensorflow.keras.models import load_model
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--slidedir", type=str,
                    help="slide dataset directory.")
parser.add_argument("--projdir", type=str,
                    help="pathaia dataset directory.")
parser.add_argument("--model", type=str,
                    help="trained model to load.")
parser.add_argument("-l", "--level", type=int,
                    default=0)
parser.add_argument("--device", default="0", type=str,
                    help="ID of the device to use for computation.")

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device


def main():
    slide_dir = args.slidedir
    proj_dir = args.projdir

    handler = data.PathaiaHandler(proj_dir, slide_dir)
    ptcs, tags = handler.list_patches(args.level, (224, 224), label='Unlabeled')
    ptcs, tags, _ = get_whole_dataset(ptcs, tags)
    preproc = xception.preprocess_input

    model = load_model(args.model)

    patches = data.get_tf_dataset(ptcs, tags, preproc, 1, 224)
    Y_pred = model.predict(
        patches,
        use_multiprocessing=True,
        workers=16
    )

    y_pred = np.argmax(Y_pred, axis=1)
    for i in range(len(y_pred)):
        ptcs[i]['Result_0'] = Y_pred[i][0]
        ptcs[i]['Result_1'] = Y_pred[i][1]

    slide_list = [p['slide'] for p in ptcs]
    for ptc_folder in tqdm(data.get_patch_folders_in_project(proj_dir)):
        print(f'Get patches from {ptc_folder}')
        patches_csv = data.get_patch_csv_from_patch_folder(ptc_folder)
        patches = pd.read_csv(patches_csv, sep=None, engine='python')
        patches = patches.set_index('id')
        slidename = os.path.basename(ptc_folder).split('_')[2]
        indeces = [i for i, x in enumerate(slide_list) if x == slidename]
        for i in tqdm(indeces):
            patches.loc[ptcs[i]['id'], 'Result_0'] = ptcs[i]['Result_0']
            patches.loc[ptcs[i]['id'], 'Result_1'] = ptcs[i]['Result_1']
        patches.to_csv(patches_csv)


if __name__ == "__main__":
    main()
