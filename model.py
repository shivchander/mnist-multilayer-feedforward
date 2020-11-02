#!/usr/bin/env python3
__author__ = "Shivchander Sudalairaj"
__license__ = "MIT"

'''
Model Definition: Multi Layer DenseNN
'''

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

np.random.seed(1)


class DenseNN(object):
    def __init__(self):
        self.n_x = 784
        self.n_h = 100
        self.n_l = 3
        self.n_y = 10
        self.layer_dims = []
        self.parameters = {}
        self.X = None
        self.y = None

    def initialize_parameters(self, n_x, n_h, n_l, n_y):
        self.n_x = n_x
        self.n_h = n_h
        self.n_l = n_l
        self.n_y = n_y
        self.layer_dims = [n_x] + [n_h] * n_l + [n_y]

        for l in range(1, len(self.layer_dims)):
            self.parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l - 1]) * 0.01
            self.parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))

            assert (self.parameters['W' + str(l)].shape == (self.layer_dims[l], self.layer_dims[l - 1]))
            assert (self.parameters['b' + str(l)].shape == (self.layer_dims[l], 1))

        return self.parameters

    def sigmoid(self, Z):
        A = 1 / (1 + np.exp(-Z))
        cache = Z

        return A, cache

    def relu(self, Z):
        A = np.maximum(0, Z)

        cache = Z
        return A, cache

    def activation_forward(self, A_prev, W, b, activation):

        def linear_forward(A, W, b):
            Z = np.dot(W, A) + b
            cache = (A, W, b)
            return Z, cache

        if activation == "sigmoid":
            Z, linear_cache = linear_forward(A_prev, W, b)
            A, activation_cache = self.sigmoid(Z)

        elif activation == "relu":
            Z, linear_cache = linear_forward(A_prev, W, b)
            A, activation_cache = self.relu(Z)

        cache = (linear_cache, activation_cache)

        return A, cache

    def forward_propagation(self, X, parameters):
        caches = []
        A = X
        L = len(parameters) // 2

        for l in range(1, L):
            A_prev = A
            A, cache = self.activation_forward(A_prev, parameters['W' + str(l)],
                                               parameters['b' + str(l)], activation='relu')
            caches.append(cache)

        AL, cache = self.activation_forward(A, parameters['W' + str(L)],
                                            parameters['b' + str(L)], activation='sigmoid')
        caches.append(cache)
        return AL, caches

    def compute_cost(self, AL, Y):
        from sklearn.metrics import log_loss
        m = Y.shape[1]
        cost = 0
        for yt, yp in zip(Y.T, AL.T):
            cost += log_loss(yt, yp)
        return cost / m

    def linear_backward(self, dZ, cache):

        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = np.dot(dZ, cache[0].T) / m
        db = np.squeeze(np.sum(dZ, axis=1, keepdims=True)) / m
        dA_prev = np.dot(cache[1].T, dZ)

        return dA_prev, dW, db

    def relu_backward(self, dA, cache):
        Z = cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0

        return dZ

    def sigmoid_backward(self, dA, cache):
        Z = cache
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)

        return dZ

    def linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache

        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            db = db.reshape(len(db), 1)

        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
            db = db.reshape(len(db), 1)

        return dA_prev, dW, db

    def backward_propagation(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        current_cache = caches[L - 1]
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL,
                                                                                                           current_cache,
                                                                                                           "sigmoid")

        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 2)], current_cache,
                                                                             "relu")
            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def initialize_velocity(self, parameters):
        L = len(parameters) // 2
        v = {}

        for l in range(L):
            v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
            v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])
        return v

    def update_parameters_with_momentum(self, parameters, grads, v, learning_rate):
        L = len(parameters) // 2
        beta = 0.9
        for l in range(L):
            # compute velocities
            v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads['dW' + str(l + 1)]
            v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads['db' + str(l + 1)]
            # update parameters
            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]

        return parameters, v

    def random_mini_batches(self, X, Y, mini_batch_size=64, seed=0):

        m = X.shape[1]
        mini_batches = []

        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((10, m))

        num_complete_minibatches = math.floor(m / mini_batch_size)
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        if m % mini_batch_size != 0:
            end = m - mini_batch_size * math.floor(m / mini_batch_size)
            mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def fit(self, X, y, n_x, n_h, n_l, n_y, learning_rate=0.001, batch_size=64, num_epochs=1000, plot_error=True):
        # parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True)

        self.X = X.T
        self.y = y.T
        self.initialize_parameters(n_x, n_h, n_l, n_y)
        errors = []
        v = self.initialize_velocity(self.parameters)

        # Optimization loop
        for i in range(num_epochs):
            minibatches = self.random_mini_batches(self.X, self.y, batch_size)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch

                # Forward propagation
                al, caches = self.forward_propagation(minibatch_X, self.parameters)

                # Compute cost
                cost = self.compute_cost(al, minibatch_Y)

                # Backward propagation
                grads = self.backward_propagation(al, minibatch_Y, caches)

                # update parameters
                self.parameters, v = self.update_parameters_with_momentum(self.parameters, v, grads, learning_rate)

            # Print the cost every 10 epoch
            if plot_error and i % 10 == 0:
                from sklearn.metrics import balanced_accuracy_score
                y_preds = self.predict(self.X.T)
                balanced_acc = balanced_accuracy_score(np.argmax(self.y.T, axis=1), np.argmax(y_preds, axis=1))
                error = 1 - balanced_acc
                print("Error after epoch %i: %f" % (i, error))
                errors.append(error)

            if error <= 0.01:
                print('Error is less than 1%. Stopping Training')
                break

        if plot_error:
            plt.plot(list(range(0, len(errors) * 10, 10)), errors)
            plt.ylabel('Error (1 - balanced acc)')
            plt.xlabel('epochs')
            plt.title('Training Error')
            plt.savefig('figs/error.pdf')
            plt.clf()

        return self.parameters

    def threshold_function(self, y_preds):
        rows, cols = y_preds.shape
        for row in range(rows):
            for col in range(cols):
                if y_preds[row, col] >= 0.75:
                    y_preds[row, col] = 1
                if y_preds[row, col] <= 0.25:
                    y_preds[row, col] = 0

        return y_preds

    def predict(self, X):
        X = X.T
        # Forward propagation
        a, caches = self.forward_propagation(X, self.parameters)

        return self.threshold_function(a.T)


def plot_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    import seaborn as sns
    df_cm = pd.DataFrame(cm, range(10), range(10))
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, annot=True)
    plt.savefig('figs/cm.pdf')