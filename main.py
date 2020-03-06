# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from collections import Counter

from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

from LRC import LinearRegressionClassifier


face_data = pd.read_csv('face_data.csv',header=None)

def get_idx(train_data_idx):
    idx = np.array(train_data_idx)
    idx = [idx+step for step in range(0, 736, 15)]
    train_data_idx = [loc for models in idx for loc in models]
    return train_data_idx

def plot_face():
    new_data = face_data.drop(225, axis=1)
    for idx in new_data.index:
        data = np.array(new_data.iloc[idx]).reshape(15,15)
        plt.imshow(data,cmap=plt.cm.gray, interpolation='nearest')
        plt.show()

#plot_face()

kf = KFold(n_splits=10, shuffle=True)
for train_data_idx, test_data_idx in kf.split(range(0,15)):
    train_data = face_data.iloc[get_idx(train_data_idx)]
    test_data = face_data.iloc[get_idx(test_data_idx)]
    clf = LinearRegressionClassifier(blocks=1)
    clf.fit(train_data)

    predict = clf.predict(test_data.drop(225, axis=1))
    test_labels = test_data[225][get_idx(test_data_idx)]

    res = Counter(map(lambda x: 1 if x[0] == x[1] else 0, zip(test_labels, predict)))
    print('precision: ', res[1] * 1.0 / (res[1] + res[0]))