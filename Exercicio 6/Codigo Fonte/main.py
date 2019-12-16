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

neural1 = MLP()
neural1.add_layer(2)
neural1.add_layer(2)
neural1.add_layer(1)
for i in range(len(train)):
    out = neural1.feed_forward(train[i])
    neural1.train(train[i], labels[i])