#!/usr/bin/env python3
__author__ = "Shivchander Sudalairaj"
__license__ = "MIT"

'''
Multi Class Classification of MNIST Dataset using Multi Layer Feed Forward Neural Net implemented from scratch
'''
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import balanced_accuracy_score
import numpy as np
from model import DenseNN, plot_confusion_matrix


def parse_data(feature_file, label_file):
    """
    :param feature_file: Tab delimited feature vector file
    :param label_file: class label
    :return: dataset as a pandas dataframe (features+label)
    """
    features = pd.read_csv(feature_file, sep="\t", header=None)
    labels = pd.read_csv(label_file, header=None)
    features['label'] = labels
    return features


def split_data(dataset):
    """
    Randomly choose 4,000 data points from the data files to form a training set, and use the remaining
    1,000 data points to form a test set. Make sure each digit has equal number of points in each set
    (i.e., the training set should have 400 0s, 400 1s, 400 2s, etc., and the test set should have 100 0s,
    100 1s, 100 2s, etc.)
    :param dataset: pandas datafrome (features+label)
    :return: None. Saves Train and Test datasets as CSV
    """
    # init empty dfs
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for i in range(0, 10):
        df = dataset.loc[dataset['label'] == i]
        train_split = df.sample(frac=0.8, random_state=200)
        test_split = df.drop(train_split.index)
        train_df = pd.concat([train_df, train_split])
        test_df = pd.concat([test_df, test_split])

    train_df.to_csv('dataset/MNIST_Train.csv', sep=',', index=False)
    test_df.to_csv('dataset/MNIST_Test.csv', sep=',', index=False)


def get_train_test(train_file, test_file):
    train = pd.read_csv(train_file, sep=",")
    y_train = train['label'].values.reshape(4000, 1)
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train)
    X_train = train.iloc[:, :-1].values

    test = pd.read_csv(test_file, sep=",")
    y_test = test['label'].values.reshape(1000, 1)
    y_test = mlb.fit_transform(y_test)
    X_test = test.iloc[:, :-1].values

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    # data = parse_data('dataset/MNISTnumImages5000_balanced.txt', 'dataset/MNISTnumLabels5000_balanced.txt')
    # split_data(data)
    # train = pd.read_csv('dataset/MNIST_Train.csv', sep=",")
    # test = pd.read_csv('dataset/MNIST_Test.csv', sep=",")

    X_train, y_train, X_test, y_test = get_train_test('dataset/MNIST_Train.csv', 'dataset/MNIST_Test.csv')
    model = DenseNN()
    model.fit(X_train, y_train, 784, 200, 2, 10, learning_rate=0.01, batch_size=32,
              num_epochs=150, plot_error=True)
    y_preds = model.predict(X_test)
    print('Accuracy: ', balanced_accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_preds, axis=1)))
    plot_confusion_matrix(y_test, y_preds)
