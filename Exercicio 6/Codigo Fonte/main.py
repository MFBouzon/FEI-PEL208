# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 19:12:20 2019

@author: Murillo

@project: PEL-208 Exercicio 6: Implementação de MLP com backpropagation
"""

from mlp import MLP
import numpy as np

from sklearn import datasets


train = []
train.append(np.array([1,1]))
train.append(np.array([1,0]))
train.append(np.array([0,1]))
train.append(np.array([0,0]))

labels = np.array([0,1,1,0])

neural1 = MLP(2)
neural1.add_layer(2)
neural1.add_layer(1)

for i in range(20000):
    for j in range(len(train)):
        neural1.train(train[j], labels[j])
        
for i in range(2):
        for j in range(2):
            out_class, out_prob = neural1.predict([i, j])
            print(out_class)
            print("Predicting XOR between {} and {} gave {} and the real is {} (Output: {:.2f})"
                  .format(i, j, out_prob > .5, bool(i) ^ bool(j), out_prob))

iris = datasets.load_iris()
names = iris.target_names
data = iris.data
labels = iris.target

neural2 = MLP(data.shape[1])
neural2.add_layer(3)
neural2.add_layer(1)

for i in range(100):
    for j in range(data.shape[0]):
        neural2.train(data[j], labels[j])
