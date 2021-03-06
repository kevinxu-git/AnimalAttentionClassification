# -*- coding: utf-8 -*-
"""
Created Jan 31 2020

@author: Guangyue
"""
import os
import rampwf as rw
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import LabelEncoder

problem_title = 'Animal Attention classification'
_target_column_name = 'class'
_prediction_label_names = [0, 1]

Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)

workflow = rw.workflows.SimplifiedImageClassifier(
    n_classes=len(_prediction_label_names),
)

score_types = [
    rw.score_types.Accuracy(name='acc', precision=3),
]


def get_cv(folder_X, y):
    _, X = folder_X
    cv = ShuffleSplit(n_splits=3, test_size=0.1, random_state=42)
    return cv.split(X, y)


def _read_data(f_name):
    data_path = os.path.join('data', '{}.csv'.format(f_name))
    data = pd.read_csv(data_path)
    X = data['id_image'].values
    Y = data['label_image'].values
    encoder = LabelEncoder()
    Y_encode = encoder.fit_transform(Y)

    return (os.path.join('data', "images"), X), Y_encode

def get_test_data(path):
    return _read_data("test")


def get_train_data(path):
    return _read_data("train")