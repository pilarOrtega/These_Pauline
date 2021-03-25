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


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('fit.log')
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
    return patches, labels


def create_model(ModelClass, n_hidden, n_classes, weights):
    base_model = ModelClass(weights=weights, include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(n_hidden, activation='relu')(x)
    predictions = Dense(n_classes, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=predictions)


def create_custom_model(ModelClass, n_hidden, n_classes, psize):
    base_model = ModelClass()
    x = Input(shape=(psize, psize, 3, ))
    y = base_model(x)
    y = GlobalAveragePooling2D()(y)
    y = Dense(n_hidden, activation='relu')(y)
    y = Dropout(0.5)(y)
    y = Dense(n_classes, activation='softmax')(y)
    return Model(inputs=x, outputs=y)


def write_experiment(filepath, task, archi, data_cfg, training_cfg, history, date):
    pretrain = training_cfg["pretrain"]
    level = data_cfg["level"]
    batch = training_cfg["batch"]
    loss = training_cfg["loss"]
    lr = training_cfg["lr"]
    bal = training_cfg["balanced"]
    da = training_cfg['data_augmentation']
    df_dict = {
        "Task": [],
        "Archi": [],
        "Pretrain": [],
        "Level": [],
        "Metric": [],
        "Run": [],
        "Epoch": [],
        "Value": [],
        "Batch": [],
        "Loss": [],
        "Lr": [],
        "Balanced": [],
        "Data_augmentation": [],
        "Date": []
    }
    for run, keras_hist in history.items():
        for metric, values in keras_hist.history.items():
            for epoch, val in enumerate(values):
                df_dict["Task"].append(task)
                df_dict["Archi"].append(archi)
                df_dict["Metric"].append(metric)
                df_dict["Run"].append(run)
                df_dict["Epoch"].append(epoch)
                df_dict["Value"].append(val)
                df_dict["Pretrain"].append(pretrain)
                df_dict["Level"].append(level)
                df_dict["Batch"].append(batch)
                df_dict["Loss"].append(loss)
                df_dict["Lr"].append(lr)
                df_dict["Balanced"].append(bal)
                df_dict["Data_augmentation"].append(da)
                df_dict["Date"].append(date)
    # put data into a dataframe
    df = pd.DataFrame()
    for name, vals in df_dict.items():
        df[name] = vals
    if os.path.exists(filepath):
        old_df = pd.read_csv(filepath)
        new_df = pd.concat([old_df, df], ignore_index=True)
        new_df.to_csv(filepath, index=False)
    else:
        df.to_csv(filepath, index=False)


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
    for task in data_cfg["tasks"]:
        logger.info("Prediction of task: '{}'".format(task))
        handler = data.PathaiaHandler(proj_dir, slide_dir)
        ptcs, tags = handler.list_patches(
            data_cfg["level"],
            (data_cfg["size"], data_cfg["size"]),
            task
        )
        patches, labels = get_whole_dataset(ptcs, tags)
        n_classes = len(np.unique(labels))
        logger.debug(
            "classes found for this task: {}".format(np.unique(labels)))
        logger.debug("counts: {}".format(
            np.unique(labels, return_counts=True)))
        splitter = StratifiedShuffleSplit(
            n_splits=experiment_cfg["folds"],
            test_size=experiment_cfg["split"],
            random_state=experiment_cfg["seed"]
        )
        # create and test different models
        logger.info("Start benchmarking")
        for name in models_CNN.models.keys():
            logger.info("Model {}".format(name))
            preproc = models_CNN.models[name]['module'].preprocess_input
            ModelClass = models_CNN.models[name]['model']
            # clear session before doing anything
            clear_session()
            # create the model
            if "custom" in name:
                model = create_custom_model(
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
                runs += 1

                # Confution Matrix and Classification Report
                test_gen_p = data.DataGenerator(
                    xtest, ytest, preproc,
                    batch_size=1,
                    dim=(data_cfg["size"], data_cfg["size"]),
                    n_channels=data_cfg["channels"],
                    n_classes=n_classes,
                    shuffle=False,
                    balanced=False,
                )
                Y_pred = model.predict(
                    test_gen_p,
                    use_multiprocessing=True,
                    workers=training_cfg["workers"]
                )
                y_pred = np.argmax(Y_pred, axis=1)
                logger.info('Confusion Matrix')
                logger.info(confusion_matrix(ytest, y_pred))
                logger.info('Classification Report')
                logger.info(classification_report(ytest, y_pred))
            outf = os.path.join(output_dir, "fit_output.csv")
            write_experiment(outf, task, name, data_cfg,
                             training_cfg, run_history, date)


if __name__ == "__main__":
    main()
