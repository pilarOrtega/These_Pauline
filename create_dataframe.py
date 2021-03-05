import pandas as pd
import glob
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--input", type=str,
                    help="WSI dataset directory.")
parser.add_argument("--outdir", type=str,
                    help="path to output patch files.")

args = parser.parse_args()


def main():
    input = args.input
    outdir = args.outdir
    df = pd.DataFrame([], columns=['Slide', 'Diagnostic', 'Annotation', 'level', 'x', 'y', 'dx', 'dy'])
    folder_list = glob.glob(os.path.join(input, '*'))
    cat_list = ['DHL_BCL2', 'DHL_BCL6', 'DLBCL_sans_rearrangement', 'THL']
    folder_list = [f for f in folder_list if os.path.basename(f) in cat_list]
    for folder in tqdm(folder_list):
        print(f'Retrieving patches from {folder}')
        diagnostic = os.path.basename(folder)
        for subfold in tqdm(glob.glob(os.path.join(folder, '*'))):
            annotation = os.path.basename(subfold)
            slide = annotation.split('_')[2]
            try:
                csv_file = os.path.join(subfold, 'patches.csv')
            except FileNotFoundError:
                print(f'No patches for {annotation}')
                continue
            patches = pd.read_csv(csv_file, sep=None, engine='python')
            for i, row in patches.iterrows():
                df = df.append({'Slide': slide,
                                'Diagnostic': diagnostic,
                                'Annotation': annotation,
                                'level': row['level'],
                                'x': row['x'],
                                'y': row['y'],
                                'dx': row['dx'],
                                'dy': 'dy'}, ignore_index=True)
    df.to_csv(os.path.join(outdir, 'all_patches.csv'), index=False)


if __name__ == "__main__":
    main()
