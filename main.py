#!/usr/bin/env python3
__author__ = "Shivchander Sudalairaj"
__license__ = "MIT"

'''
Multi Class Classification of MNIST Dataset using Multi Layer Feed Forward Neural Net implemented from scratch
'''

from utils import *
from model import DenseNN, plot_confusion_matrix
from autoencoder import AutoencoderNN


if __name__ == '__main__':
    # data = parse_data('dataset/MNISTnumImages5000_balanced.txt', 'dataset/MNISTnumLabels5000_balanced.txt')
    # split_data(data)
    # train = pd.read_csv('dataset/MNIST_Train.csv', sep=",")
    # test = pd.read_csv('dataset/MNIST_Test.csv', sep=",")

    # # Classification
    X_train, y_train, X_test, y_test = get_train_test('dataset/MNIST_Train.csv', 'dataset/MNIST_Test.csv')
    # model = DenseNN()
    # model.fit(X_train, y_train, 784, 200, 1, 10, learning_rate=0.01, batch_size=32,
    #           num_epochs=150, plot_error=True)
    # y_preds = model.predict(X_test)
    # print('Accuracy: ', balanced_accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_preds, axis=1)))
    # plot_confusion_matrix(y_test, y_preds)

    # # Autoencoder

    model2 = AutoencoderNN()
    model2.fit(X_train, X_train, 784, 200, 1, 784, learning_rate=0.01, batch_size=32, num_epochs=150, plot_error=True)
    random_outputs(model2, X_test)
    plot_train_test_error(model2, X_train, X_test)
    train_test_digit_error(model2, X_train, X_test)
