import scipy.io as scio
import torch
import torch.utils.data as Data
from sklearn.model_selection import StratifiedShuffleSplit

DATA_DIR = ''


def queryData(file):
    x_train, x_test, y_train, y_test = queryDataFromMat(file)
    return normalData(x_train), normalData(x_test), normalData(y_train), normalData(y_test)


def queryDataFromMat(file, test_size=0.4):
    datafile = DATA_DIR + 'data\\dataset\\' + file
    data = scio.loadmat(datafile)
    x = data['X']
    y = data['y']

    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=98765)

    for train_index, test_index in split.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        return x_train, x_test, y_train, y_test


def normalData(x):
    x = torch.from_numpy(x).to(torch.float32).cuda()
    return x


def getDataLoader(file, batchsize=2, shuffle=True):
    x_train, x_test, y_train, y_test = queryData(file)
    trainSet = Data.TensorDataset(x_train, y_train)
    testSet = Data.TensorDataset(x_test, y_test)
    trainLoader = Data.DataLoader(dataset=trainSet,
                                  batch_size=batchsize,
                                  shuffle=shuffle)
    testLoader = Data.DataLoader(dataset=testSet,
                                 batch_size=batchsize,
                                 shuffle=shuffle)
    return trainLoader, testLoader, x_train, x_test, y_train, y_test
