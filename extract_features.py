import os
import pandas as pd
import argparse
from Pathaia_CNN_embedding import get_embeddings
import pickle


def select_patches(patches, level, patch_size, slide_dir):
    patch_list = []
    patches = patches[patches['level'] == level]
    for i, row in patches.iterrows():
        slide_path = os.path.join(slide_dir, row['Diagnostic'])
        slide_path = os.path.join(slide_path, row['Annotation'])+'.mrxs'
        patch_list.append(
            {'index': i,
             'slide': slide_path,
             'x': row['x'],
             'y': row['y'],
             'level': row['level'],
             'dimensions': (patch_size, patch_size),
             'true_label': row['Diagnostic']}
        )
    return patch_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patches', type=str)
    parser.add_argument('--slide_dir', type=str)
    parser.add_argument('--level', type=int, default=0)
    parser.add_argument('--patch_size', type=int, default=224)
    parser.add_argument('--model', type=str, default='ResNet50')
    parser.add_argument('--layer', type=str, default='')
    parser.add_argument('--outdir', type=str, default='')
    args = parser.parse_args()

    patches = args.patches
    slide_dir = args.slide_dir
    level = args.level
    patch_size = args.patch_size
    model = args.model
    layer = args.layer
    outdir = args.outdir

    patches = pd.read_csv(patches, sep=None, engine='python')
    patch_list = select_patches(patches, level, patch_size, slide_dir)
    patch_list = dict(enumerate(patch_list))
    descriptors = get_embeddings(model, patch_list, patch_size=patch_size, layer=layer)
    for p in patch_list:
        patch_list[p][f'Descriptor_{model}'] = descriptors[p]
    if outdir != '':
        with open(os.path.join(outdir, f'patches_{model}_level{level}.p'), 'wb') as f:
            pickle.dump(patch_list, f)


if __name__ == "__main__":
    main()
