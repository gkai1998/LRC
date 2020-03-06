# -*- coding: utf-8 -*-


from numpy import array,dot
from numpy.linalg import norm,pinv

class LinearRegression:

    def __init__(self, train_data, blocks=1):
        self.weights = list()
        self.blocks = blocks
        train_data = self.get_data(train_data, self.blocks)

        for data in train_data:
            X = array(data)
            Y = X
            A = dot(X.T, X)
            B = dot(X.T, Y)
            # print(pinv(A).dot(B).shape)
            self.weights.append(pinv(A).dot(B))

    def predict(self, test_data):
        test_data = self.get_data(test_data, self.blocks)
        blocks = list()

        for data, weights in zip(test_data, self.weights):
            Y = array(data)
            re_Y = dot(Y, weights)
            blocks.append([norm(Y[idx]-re_Y[idx],2) for idx in range(len(data))])

        return [min(map(lambda x: x[idx], blocks)) for idx in range(len(test_data[0]))]

    def get_data(self, data, blocks):
        split_list = lambda _list, n: [_list[i * n:(i + 1) * n] for i in range(int((len(_list) + n - 1) / n))]

        columns = split_list(list(data.columns), int(len(data.columns)/blocks))
        print(columns)
        return [data.ix[:,col] for col in columns]


class LinearRegressionClassifier:

    def __init__(self, blocks=1):
        self.clfs = list()
        self.blocks = blocks

    def fit(self, train_data):
        for models in train_data[225].unique():
            data = train_data[train_data[225]==models]
            self.clfs.append(LinearRegression(data.drop(225, axis=1), blocks=self.blocks))

    def predict(self, test_data):
        dis = [clf.predict(test_data) for clf in self.clfs]

        res = list()
        for idx, _ in enumerate(array(test_data)):
            data_pre_list = list(map(lambda x: x[idx], dis))
            res.append(data_pre_list.index(min(data_pre_list)) + 1)
        return array(res)