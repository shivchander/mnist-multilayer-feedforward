#!/usr/bin/env python3
__author__ = "Shivchander Sudalairaj"
__license__ = "MIT"

'''
Model Definition: Multi Layer Feed Forward (Single Hidden Layer)
'''

import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class FeedForwardNN:
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        self.n_x = input_layer_size
        self.n_h = hidden_layer_size
        self.n_y = output_layer_size
        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None
        self.init_parameters()

    def init_parameters(self):
        self.w1 = np.random.randn(self.n_h, self.n_x) * 0.01
        self.b1 = np.zeros(shape=(self.n_h, 1))
        self.w2 = np.random.randn(self.n_y, self.n_h) * 0.01
        self.b2 = np.zeros(shape=(self.n_y, 1))

    def forward_prop(self, X):
        # shape of X should be (n_x, m)
        z1 = np.dot(self.W1, X) + self.b1
        a1 = np.tanh(z1)
        z2 = np.dot(self.W2, a1) + self.b2
        a2 = sigmoid(z2)

        cache = {"z1": z1,
                 "a1": a1,
                 "z2": z2,
                 "a2": a2}

        return a2, cache

    def back_prop(self, X, y, cache):
        # shape of X should be (n_x, m)
        # shape of y should be (1, m)
        a1 = cache['a1']
        a2 = cache['a2']
        m = X.shape[1]

        dz2 = a2 - y
        dw2 = (1 / m) * np.dot(dz2, a1.T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
        dz1 = np.multiply(np.dot(self.w2.T, dz2), 1 - np.power(a1, 2))
        dw1 = (1 / m) * np.dot(dz1, X.T)
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

        grads = {"dw1": dw1,
                 "db1": db1,
                 "dw2": dw2,
                 "db2": db2}

        return grads

    def update_params(self, grads, v, beta, lr):















    # def cost(self, y_hat, y):
    #     m = y.shape[1]
    #     logprobs = np.multiply(np.log(y_hat), y) + np.multiply((1 - y), np.log(1 - y_hat))
    #     cost = - np.sum(logprobs) / m
    #     cost = np.squeeze(cost)
    #
    #     return cost

    def back_prop(self, cost):





