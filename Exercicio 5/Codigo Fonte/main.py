# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 00:18:28 2019

@author: Murillo

@project: PEL-208 Exercicio 5: Implementação de um Perceptron
"""

import numpy as np
from perceptron import Perceptron
import plot
from sklearn import datasets

#Entrada do primeiro teste
train = []
train.append(np.array([1,1]))
train.append(np.array([1,0]))
train.append(np.array([0,1]))
train.append(np.array([0,0]))

names = np.array(["False", "True"])

#Teste AND
labels = np.array([1,0,0,0])

p1 = Perceptron(train[0].shape[0])
p1.train(train, labels)
result = []

for i in train:
    result.append(p1.predict(i))
result = np.asarray(result)
plot.perceptron_result(train, result, "Result AND", names)

#Teste OR
labels = np.array([1,1,1,0])

p1 = Perceptron(train[0].shape[0])
p1.train(train, labels)
result = []

for i in train:
    result.append(p1.predict(i))
result = np.asarray(result)
plot.perceptron_result(train, result, "Result OR", names)

#Teste XOR
labels = np.array([0,1,1,0])

p1 = Perceptron(train[0].shape[0])
p1.train(train, labels)
result = []

for i in train:
    result.append(p1.predict(i))
result = np.asarray(result)
plot.perceptron_result(train, result, "Result XOR", names)


#Leitura da base iris
iris = datasets.load_iris()
names = iris.target_names
data = iris.data
labels = iris.target

newData = []
for i in range(len(data)):
    if labels[i] != 2:
        newData.append(data[i])

p2 = Perceptron(newData[0].shape[0])
p2.train(newData, labels)
result = []
for i in newData:
    result.append(p2.predict(i))
plot.perceptron_result(newData, result, "Iris classification Setosa x Versicolor", names)