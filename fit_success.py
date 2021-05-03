import argparse
import os
import yaml
import pandas as pd
from auxiliary_functions import get_whole_dataset
import numpy as np
from models import models_CNN
from sklearn.model_selection import StratifiedShuffleSplit
from keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
import logging
import data
import fit
from sklearn.metrics import classification_report, confusion_matrix


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('fit_success.log')
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

parser.add_argument("--config", type=str,
                    help="path to a config file.")
parser.add_argument("--device", default="0", type=str,
                    help="ID of the device to use for computation.")

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device


def list_patch(df, task):
    df = df[df['Task'] == task]
    ptcs, labels = [], []
    for _, row in df.iterrows():
        patch = {"x": row["X"],
                 "y": row["Y"],
                 "level": row["Level"],
                 "slide_path": row["Slide"],
                 "slide_name": os.path.basename(row["Slide"])}
        ptcs.append(patch)
        labels.append(row["Success"])
    return ptcs, labels


def main():
    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    date = cfg["date"]
    data_cfg = cfg["data"]
    input = cfg["input"]
    training_cfg = cfg["training"]
    experiment_cfg = cfg["experiment"]
    archi_cfg = cfg["architecture"]

    patches_csv = input['patches']
    tasks = data_cfg['tasks']
    output_dir = input['output_dir']
    exp_date = input['date']

    patches_csv = pd.read_csv(patches_csv, sep=None, engine='python')
    patches_csv = patches_csv[patches_csv['Date'] == exp_date]
    for task in tasks:
        ptcs, tags = list_patch(patches_csv, task)
        patches, labels, labels_dict = get_whole_dataset(ptcs, tags)
        n_classes = len(np.unique(labels))
        logger.debug(
            "classes found for this task: {}".format(np.unique(labels)))
        logger.debug(
            "classes: {}".format(labels_dict)
        )
        logger.debug("counts: {}".format(
            np.unique(labels, return_counts=True)))
        splitter = StratifiedShuffleSplit(
            n_splits=experiment_cfg["folds"],
            test_size=experiment_cfg["split"],
            random_state=experiment_cfg["seed"]
        )
        # create and test different models
        for name in archi_cfg['archs']:
            preproc = models_CNN.models[name]['module'].preprocess_input
            ModelClass = models_CNN.models[name]['model']
            # clear session before doing anything
            clear_session()
            # create the model
            if "custom" in name:
                model = fit.create_custom_model(
                    ModelClass,
                    archi_cfg["hidden"],
                    len(np.unique(labels)),
                    data_cfg["size"]
                )
            else:
                model = fit.create_model(
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

            run_history = dict()
            runs = 1
            # train and validate the model
            slides = [x['slide_name'] for x in patches]
            # get name slide
            slides = [s.split('_')[2] for s in slides]
            slides, indices = np.unique(slides, return_index=True)
            labels_slides = [x for x in labels[indices]]
            results = []
            for train_indices, test_indices in splitter.split(slides, labels_slides):
                train_slides, test_slides = slides[train_indices], slides[test_indices]
                xtrain, xtest, ytrain, ytest = [], [], [], []
                for i in range(len(patches)):
                    if patches[i]['slide_name'].split('_')[2] in train_slides:
                        xtrain.append(patches[i])
                        ytrain.append(labels[i])
                    elif patches[i]['slide_name'].split('_')[2] in test_slides:
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
                for i in range(len(xtest)):
                    xtest[i]['True'] = ytest[i]
                    xtest[i]['Prediction'] = y_pred[i]
                    xtest[i]['Run'] = runs
                results.extend(xtest)
                logger.info('Classification Report')
                logger.info(classification_report(ytest, y_pred))
                runs += 1
            outf = os.path.join(output_dir, "fit_success_output.csv")
            fit.write_experiment(outf, task, name, data_cfg,
                                 training_cfg, run_history, date)
            outf = os.path.join(output_dir, f"results_success_{name}.csv")
            fit.write_results(outf, task, name, data_cfg,
                              training_cfg, results, date)


if __name__ == "__main__":
    main()
