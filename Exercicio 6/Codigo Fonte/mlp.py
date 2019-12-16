# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 19:13:01 2019

@author: Murillo


@project: PEL-208 Exercicio 6: Implementação de MLP com backpropagation
"""

import numpy as np
import math
import operator

class MLP:
    def __init__(self, n_inputs, learning_rate = 0.1):
        self.layer_size = []
        self.layer_size.append(n_inputs)
        self.weights = []
        self.biases = []
        self.functions = []
        self.learning_rate = learning_rate
        
    def add_layer(self, n, function="sigmoid"):
        self.layer_size.append(n)
        self.weights.append(np.zeros(n))
        self.functions.append(function)
        self.biases.append(0)
    
    @staticmethod
    def relu(x):
        rel = np.vectorize(lambda y: max(0, y))
        return rel(x)
    
    @staticmethod
    def sigmoid(x):
        sig = np.vectorize(lambda y:  (1 - 1 / (1 + math.exp(y))) if y < 0 else  (1 / (1 + math.exp(-y))))
        return sig(x)
    
    @staticmethod
    def squash(x, function):
        if function == "sigmoid":
            return MLP.sigmoid(x)
        elif function == "relu":
            return MLP.relu(x)
        
    @staticmethod
    def derivative(x, function):
        if function == "sigmoid":
            return np.multiply(x, (1-x))
        elif function == "relu":
            d_relu = np.vectorize(lambda y: 1 if y > 0 else 0)
            return d_relu(x)
    
    
    def feed_forward(self, x):
        out = [np.matrix(x).T]
        
        for i in range(len(self.layer_size)-1):
            out.append(MLP.squash(np.dot(self.weights[i], out[-1]) + self.biases[i], self.functions[i]))
        
        return out
    
    def train(self, x, labels):
        labels = np.matrix(labels).T
        out = self.feed_forward(x)
        print(out)
        errors = [np.subtract(labels, out[-1])]
        for i in range(len(self.weights) - 1):
            errors.insert(0, np.dot(self.weights[-1-i].T, errors[0]))
        
        for i in range(len(self.weights)):
            gradient = np.multiply(errors[-1-i], MLP.derivative(out[-1-i], self.functions[-1-i]))
            gradient *= self.learning_rate
            self.biases[-1-i] += gradient
            delta_w = np.dot(gradient, out[-2-i].T)
            self.weights[-1-i] += delta_w
            
    def predict(self, x):
        out = self.feed_forward(x)[-1]
        out = dict(enumerate(out.A1))
        out_class = max(out.items(), key=operator.itemgetter(1))[0]
        out_prob = out[out_class]
        
        return out_class, out_prob