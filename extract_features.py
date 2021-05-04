import os
import argparse
import tensorflow as tf
import pathaia.util.management as util


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proj_dir', type=str)
    parser.add_argument('--slide_dir', type=str)
    parser.add_argument('--patch_size', type=int, default=224)
    parser.add_argument('--model', type=str, default='ResNet50')
    parser.add_argument('--layer', type=str, default='')
    parser.add_argument('--outdir', type=str, default='')
    parser.add_argument('--device', default="0")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    proj_dir = args.patches
    slide_dir = args.slide_dir
    level = args.level
    patch_size = args.patch_size
    model = args.model
    layer = args.layer

    handler = util.PathaiaHandler(proj_dir, slide_dir)
    handler.extract_features(model=model, patch_size=patch_size, level=level,
                             layer=layer)


if __name__ == "__main__":
    main()
