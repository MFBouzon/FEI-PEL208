# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 15:30:47 2019

@author: Murillo


@project: PEL-208 Exercicio 4: Implementação do algoritmo de clusterização K-Means

"""

import statistic
import pandas as pd
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np



#bases
files = ["wine.csv", "divorce.csv"]

#leitura da base de exemplo
data = pd.read_csv("xclara.csv", sep=',')
data.dropna(inplace = True)
data = data.to_numpy()

statistic.kmeans(data, 3)


#leitura da base do exercicio 1
data = pd.read_csv("data1.csv", sep=';', header = None)
data.dropna(inplace = True)
data = data.to_numpy()

statistic.kmeans(data, 3)

#Leitura da base iris
iris = datasets.load_iris()
names = iris.target_names
data = iris.data
labels = iris.target

clusters = statistic.kmeans(data, 3)
print(clusters)

for i in range(3):
    print(names[i])
    print(sum(clusters == i))
    print(sum(labels == i))
    print(min(sum(clusters == i),sum(labels == i))/max(sum(clusters == i),sum(labels == i)) * 100, "%")
    
#calcular o K-means para cada base do exercicio 2
for fname in files:
    #leitura da base
    data = pd.read_csv(fname, sep=';', header=None)
    data.dropna(inplace = True)
    data = data.to_numpy()
    labels = pd.read_csv(fname[:len(fname)-3] + "target")
    labels = labels.to_numpy()
    classes = np.unique(labels)
    k = len(classes)
    
    clusters = statistic.kmeans(data, k)
    print(clusters)
    for i in range(k):
        print(classes[i])
        print(sum(clusters == i))
        print(sum(labels == classes[i]))
        print(min(sum(clusters == i),sum(labels == classes[i]))/max(sum(clusters == i),sum(labels == classes[i])) * 100, "%")
        