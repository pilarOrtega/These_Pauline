import data
from models import models_CNN
import os
import argparse
import logging
import pandas as pd
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.backend import clear_session
from tensorflow.keras import Model
from sklearn.model_selection import StratifiedShuffleSplit
from keras.optimizers import Adam
import numpy as np
import yaml
from tensorflow.keras.layers import Input
from sklearn.metrics import classification_report, confusion_matrix
from fit import get_whole_dataset, create_custom_model, create_model
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('WSI_fit.log')
c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.DEBUG)
c_format = logging.Formatter('%(name)s - [%(levelname)s] - %(message)s')
f_format = logging.Formatter(
    '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)
logger.addHandler(c_handler)
logger.addHandler(f_handler)

parser = argparse.ArgumentParser()

parser.add_argument("--slidedir", type=str,
                    help="slide dataset directory.")
parser.add_argument("--projdir", type=str,
                    help="pathaia dataset directory.")
parser.add_argument("--config", type=str,
                    help="path to a config file.")
parser.add_argument("--output", type=str,
                    help="path to output folder.")
parser.add_argument("--device", default="0", type=str,
                    help="ID of the device to use for computation.")

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device


divergence_fn = lambda q,p,_:tfd.kl_divergence(q,p)/226214


def create_bayesian_custom_model(ModelClass, n_hidden, n_classes, psize):
    base_model = ModelClass()
    x = tfpl.Convolution2DReparameterization(input_shape=(psize, psize, 3, ), filters=8, kernel_size=16, activation='relu',
                                             kernel_prior_fn = tfpl.default_multivariate_normal_fn,
                                             kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                                             kernel_divergence_fn = divergence_fn,
                                             bias_prior_fn = tfpl.default_multivariate_normal_fn,
                                             bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                                             bias_divergence_fn = divergence_fn)
    y = base_model(x)
    y = GlobalAveragePooling2D()(y)
    y = Dense(n_hidden, activation='relu')(y)
    y = Dropout(0.5)(y)
    y = tfpl.DenseReparameterization(units=tfpl.OneHotCategorical.params_size(n_classes), activation=None,
                                     kernel_prior_fn = tfpl.default_multivariate_normal_fn,
                                     kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                                     kernel_divergence_fn = divergence_fn,
                                     bias_prior_fn = tfpl.default_multivariate_normal_fn,
                                     bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                                     bias_divergence_fn = divergence_fn
                                     )(y)
    y = tfpl.OneHotCategorical(n_classes)(y)
    return Model(inputs=x, outputs=y)


def main():
    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    slide_dir = args.slidedir
    proj_dir = args.projdir
    output_dir = args.output

    date = cfg["date"]
    data_cfg = cfg["data"]
    training_cfg = cfg["training"]
    experiment_cfg = cfg["experiment"]
    archi_cfg = cfg["architecture"]
    logger.info(
        f"Conf data - level: {data_cfg['level']}, size: {data_cfg['size']} ")
    logger.info(
        f"Conf training - batch: {training_cfg['batch']}, epochs: {training_cfg['epochs']}, lr: {training_cfg['lr']}, loss: {training_cfg['loss']}, workers: {training_cfg['workers']}, balanced: {training_cfg['balanced']}, data_augmentation: {training_cfg['data_augmentation']}")
    logger.info(
        f"Conf experiments - folds: {experiment_cfg['folds']}, split: {experiment_cfg['split']}")

    logger.info("Prediction of tumor areas")
    # Fit network for tissue separation
    if len(data_cfg["tasks"]) == 1:
        task = data_cfg["tasks"][0]
    else:
        task = "Task_5"
    handler = data.PathaiaHandler(proj_dir, slide_dir)
    ptcs, tags = handler.list_patches(
        data_cfg["level"],
        (data_cfg["size"], data_cfg["size"]),
        task
    )
    patches, labels, label_dict = get_whole_dataset(ptcs, tags)
    logger.debug(
        "Classes: {}".format(label_dict)
    )
    n_classes = len(np.unique(labels))
    logger.debug("counts: {}".format(
        np.unique(labels, return_counts=True)))

    splitter = StratifiedShuffleSplit(
        n_splits=experiment_cfg["folds"],
        test_size=experiment_cfg["split"],
        random_state=experiment_cfg["seed"]
    )

    name = 'customLenet'
    preproc = models_CNN.models[name]['module'].preprocess_input
    ModelClass = models_CNN.models[name]['model']
    # clear session before doing anything
    clear_session()
    # create the model
    if "custom" in name:
        model = create_bayesian_custom_model(
            ModelClass,
            archi_cfg["hidden"],
            len(np.unique(labels)),
            data_cfg["size"]
        )
    else:
        model = create_model(
            ModelClass,
            archi_cfg["hidden"],
            len(np.unique(labels)),
            weights=training_cfg["pretrain"]
        )
    # create the optimizer
    opt = Adam(learning_rate=training_cfg["lr"])
    # compile model with optimizer
    model.compile(
        optimizer=opt,
        loss=training_cfg["loss"],
        metrics=["accuracy"]
    )
    logger.info(
        "Running {} fit-test procedures".format(
            experiment_cfg["folds"])
    )
    run_history = dict()
    runs = 1
    # train and validate the model
    slides = [x['slide'] for x in patches]
    slides, indices = np.unique(slides, return_index=True)
    labels_slides = [x for x in labels[indices]]
    for train_indices, test_indices in splitter.split(slides, labels_slides):
        train_slides, test_slides = slides[train_indices], slides[test_indices]
        xtrain, xtest, ytrain, ytest = [], [], [], []
        for i in range(len(patches)):
            if patches[i]['slide'] in train_slides:
                xtrain.append(patches[i])
                ytrain.append(labels[i])
            elif patches[i]['slide'] in test_slides:
                xtest.append(patches[i])
                ytest.append(labels[i])
            else:
                logger.info(
                    f"{patches[i]['slide']} not in train/test partition!!"
                )
        logger.info(
            "Run {}/{}".format(runs, experiment_cfg["folds"])
        )
        logger.info(
            f"Train slides: {len(train_slides)} - Test slides: {len(test_slides)}"
        )
        logger.debug(
            f"Train patches: {len(xtrain)} - Test patches: {len(xtest)}"
        )
        logger.debug(
            f"Train patches: {len(ytrain)} - Test patches: {len(ytest)}"
        )
        # create data generators
        train_gen = data.DataGenerator(
            xtrain, ytrain, preproc,
            batch_size=training_cfg["batch"],
            dim=(data_cfg["size"], data_cfg["size"]),
            n_channels=data_cfg["channels"],
            n_classes=n_classes,
            shuffle=True,
            balanced=training_cfg['balanced'],
            data_augmentation=training_cfg['data_augmentation']
        )
        test_gen = data.DataGenerator(
            xtest, ytest, preproc,
            batch_size=training_cfg["batch"],
            dim=(data_cfg["size"], data_cfg["size"]),
            n_channels=data_cfg["channels"],
            n_classes=n_classes,
            shuffle=True,
            balanced=False,
        )
        fit_history = model.fit(
            train_gen,
            validation_data=test_gen,
            epochs=training_cfg["epochs"],
            use_multiprocessing=True,
            workers=training_cfg["workers"]
        )
        run_history[runs] = fit_history
        model.save(os.path.join(output_dir, 'model_tumor_detect'))
    # Predict patches with tumor
    # Create patch mask for patches of interest (heatmap)
    #


if __name__ == "__main__":
    main()
