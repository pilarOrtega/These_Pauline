import pathaia.util.management as util
import numpy as np
from auxiliary_functions import get_whole_dataset
import os
import pandas as pd
from tqdm import tqdm
import numpy
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--slidedir", type=str,
                    help="slide dataset directory.")
parser.add_argument("--projdir", type=str,
                    help="pathaia dataset directory.")
parser.add_argument("--outdir", type=str)
parser.add_argument("--old_mode", default=False)
parser.add_argument("--tasks", nargs='+',
                    help="Tasks to analyze")
parser.add_argument("--levels", nargs='+', type=int,
                    help="Levels to analyze")
parser.add_argument("--date", type=str, default="")
parser.add_argument("--device", default="0", type=str,
                    help="ID of the device to use for computation.")

args = parser.parse_args()


def balanced_set(patches, labels, patch_array, replacement=True):
    tree = {}
    patch_slides = [x['slide'] for x in patches]
    patch_slides = np.asarray(patch_slides)
    classes = np.unique(labels)
    for cl in np.arange(len(classes)):
        tree[cl] = {}
        slides = [patches[i]['slide'] for i in range(len(labels)) if labels[i] == cl]
        for slide in np.unique(slides):
            tree[cl][slide] = (np.argwhere(patch_slides == slide).squeeze(1).tolist())
    n_slides = [len(tree[c]) for c in tree.keys()]
    num_samples = 0
    for c in tree.keys():
        n_patches = [len(tree[c][s]) for s in tree[c].keys()]
        for i in range(min(n_slides)):
            num_samples += min(500, n_patches[i])

    balanced_patches = []
    for _ in range(num_samples):
        x = np.random.uniform(size=3)
        classes = list(tree.keys())
        cl = classes[int(x[0]*len(classes))]
        cl_slides = tree[cl]
        slides = list(cl_slides.keys())
        slide = slides[int(x[1]*len(slides))]
        slide_patches = cl_slides[slide]
        idx = int(x[2]*len(slide_patches))
        if replacement:
            patch = slide_patches[idx]
        else:
            patch = slide_patches.pop(idx)
            if len(slide_patches) == 0:
                cl_slides.pop(slide)
                if len(cl_slides) == 0:
                    tree.pop(cl)
        balanced_patches.append(patch)

    patches_mod = []
    labels_mod = []
    patch_array_mod = []
    for x in balanced_patches:
        patches_mod.append(patches[x])
        labels_mod.append(labels[x])
        patch_array_mod.append(patch_array[x])
    return patches_mod, labels_mod, patch_array_mod


def main():
    proj_dir = args.projdir
    slide_dir = args.slidedir
    method = 'ResNet50'
    tasks = args.tasks
    levels = args.levels
    outdir = args.outdir
    date = args.date
    old_mode = args.old_mode
    handler = util.PathaiaHandler(proj_dir, slide_dir)

    names = ["Nearest Neighbors k=1",
             "Linear Regression"
             # "Linear SVM",
             # "Decision Tree",
             # "Random Forest",
             # "Neural Net",
             # "AdaBoost",
             # "Naive Bayes",
             # "QDA"
             ]

    classifiers = [
        KNeighborsClassifier(1),
        LogisticRegression(max_iter=1000, multi_class='auto'),
        # SVC(kernel="linear", C=0.025, probability=True),
        # DecisionTreeClassifier(),
        # RandomForestClassifier(n_estimators=100),
        # MLPClassifier(max_iter=1000),
        # AdaBoostClassifier(),
        # GaussianNB(),
        # QuadraticDiscriminantAnalysis()
        ]



    for t in tasks:
        for level in levels:
            level = int(level)
            file = f'/data/Projet_Pauline/{t}_level{level}.npy'
            patches, labels = handler.list_patches(level=level, dim=(224, 224), column=t)
            if old_mode:
                labels = [labels[i] for i in range(len(labels)) if not 'Cas_supplementaires' in patches[i]['slide_path']]
                patches = [p for p in patches if not 'Cas_supplementaires' in p['slide_path']]
                for i in range(len(patches)):
                    if 'Cas_supplementaires' in patches[i]['slide_path']:
                        del patches[i]
                        del labels[i]
            if t in ['Task_1', 'Task_2', 'Task_3']:
                labels = ['NR' if v == 'NR' else 'R' if v == 'R' else 'NA' for v in labels]
            elif t in ['Task_5']:
                labels = ['T' if v == 'T' else 'N' if v == 'N' else 'NA' for v in labels]
            patches, labels, labels_dict = get_whole_dataset(patches, labels)
            inv_labels_dict = {v: k for k, v in labels_dict.items()}
            if os.path.exists(file) and not old_mode:
                patch_array = np.load(file)
                print(f'Loaded file {file}')
            else:
                print(f'Extracting {t} from level {level}')
                patch_array = np.zeros((len(patches), 512))
                x = 0
                for patch in tqdm(patches):
                    folder = os.path.dirname(patch['slide']).replace(slide_dir, proj_dir)
                    folder = os.path.join(folder, patch['slide_name'].split('.')[0])
                    feat_file = os.path.join(folder, f'features_{method}.csv')
                    df = pd.read_csv(feat_file)
                    df = df.set_index('id')
                    features = df.loc[patch['id']]
                    for y in range(512):
                        patch_array[x, y] = features[f'{y}']
                    x += 1
                if not old_mode:
                    np.save(f'/data/Projet_Pauline/{t}_level{level}.npy', patch_array)
            # train and validate the model
            slides = [x['slide_name'] for x in patches]
            # get name slide
            slides = [s.split('_')[2] for s in slides]
            slides, indices = np.unique(slides, return_index=True)
            labels_slides = [x for x in labels[indices]]
            labels_slides = np.asarray(labels_slides)
            try:
                df = pd.read_csv(os.path.join(outdir, f'Slide_pred_{t}_level{level}_{date}.csv'), sep=None, engine='python')
            except:
                df = pd.DataFrame([], columns=['Slide',
                                               'Method',
                                               'Task',
                                               'Date',
                                               'Level',
                                               'True',
                                               'Fold',
                                               'Predict_0',
                                               'Predict_1',
                                               'Avg_0',
                                               'Avg_1'])
            for name, model_func in zip(names, classifiers):
                print(f'Evaluating classifier {name}')
                splitter = StratifiedKFold(
                    n_splits=5,
                    shuffle=True,
                    random_state=42
                )
                fold = 0
                scores = []
                for train_indices, test_indices in splitter.split(slides, labels_slides):
                    model = model_func
                    train_slides, test_slides = slides[train_indices], slides[test_indices]
                    train_labels, test_labels = labels_slides[train_indices], labels_slides[test_indices]
                    xtrain, xtest, ytrain, ytest, train_patches, test_patches = [], [], [], [], [], []
                    for i in range(len(patches)):
                        if patches[i]['slide_name'].split('_')[2] in train_slides:
                            xtrain.append(patch_array[i, :])
                            ytrain.append(labels[i])
                            train_patches.append(patches[i])
                        elif patches[i]['slide_name'].split('_')[2] in test_slides:
                            xtest.append(patch_array[i, :])
                            ytest.append(labels[i])
                            test_patches.append(patches[i])
                    print('Balance set...')
                    train_patches, ytrain, xtrain = balanced_set(train_patches, ytrain, xtrain)
                    print(f'Rebalanced: {np.unique(ytrain, return_counts=True)}')
                    print('Start fitting...')
                    model.fit(xtrain, ytrain)
                    score = model.score(xtest, ytest)
                    scores.append(score)
                    predictions = model.predict(xtest)
                    predictions_proba = model.predict_proba(xtest)
                    print('Accuracy for fold {}: {}'.format(i, score))
                    table = classification_report(
                        ytest, predictions, target_names=list(labels_dict.keys()))
                    print(table)
                    with open(os.path.join(outdir, f'{name}_{t}_level{level}_fold{fold}.txt'), 'w') as f:
                        f.write(table)
                    # Save predictions to PathAIA folder:
                    pathaia_folders = [x['slide'] for x in test_patches]
                    pathaia_folders = np.unique(pathaia_folders)
                    for folder in pathaia_folders:
                        df_pathaia_folder = util.get_patch_csv_from_patch_folder(folder.replace(slide_dir, proj_dir).split('.')[0])
                        df_pathaia = pd.read_csv(df_pathaia_folder, sep=None, engine='python')
                        df_pathaia = df_pathaia.set_index('id')
                        for i in range(len(test_patches)):
                            if test_patches[i]['slide'] == folder:
                                df_pathaia.loc[test_patches[i]['id'], f'Pred_{name}_{t}'] = inv_labels_dict[predictions[i]]
                                df_pathaia.loc[test_patches[i]['id'], f'Prob_{name}_{t}_0'] = round(predictions_proba[i, 0], 2)
                                df_pathaia.loc[test_patches[i]['id'], f'Prob_{name}_{t}_1'] = round(predictions_proba[i, 1], 2)
                        df_pathaia = df_pathaia.reset_index()
                        df_pathaia.to_csv(df_pathaia_folder, index=False)
                    fold += 1
                    for x in range(len(test_slides)):
                        slide, label = test_slides[x], test_labels[x]
                        results, prob_0, prob_1 = [], [], []
                        for i in range(len(test_patches)):
                            if test_patches[i]['slide_name'].split('_')[2] == slide:
                                results.append(predictions[i])
                                prob_0.append(predictions_proba[i, 0])
                                prob_1.append(predictions_proba[i, 1])
                        predict_0 = results.count(0)
                        predict_1 = results.count(1)
                        avg_0 = sum(np.asarray(prob_0))/len(prob_0)
                        avg_1 = sum(np.asarray(prob_1))/len(prob_1)
                        df = df.append({'Slide': slide,
                                        'Method': name,
                                        'Task': t,
                                        'Date': date,
                                        'Level': level,
                                        'True': label,
                                        'Fold': fold,
                                        'Predict_0': predict_0,
                                        'Predict_1': predict_1,
                                        'Avg_0': avg_0,
                                        'Avg_1': avg_1}, ignore_index=True)
                    df['Ratio'] = df['Predict_1']/(df['Predict_1']+df['Predict_0'])
                    df.to_csv(os.path.join(outdir, f'Slide_pred_{t}_level{level}_{date}.csv'), index=False)


if __name__ == "__main__":
    main()
