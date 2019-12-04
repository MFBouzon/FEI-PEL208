# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 15:30:47 2019

@author: Murillo


@project: PEL-208 Exercicio 4: Implementação algoritmo de clusterização K-Means

"""

import statistic
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import numpy as np

#plt.rcParams['figure.figsize'] = (10, 6)
#plt.style.use('ggplot')

#bases
files = ["xclara.csv"]

#calcular o K-means para cada base
for fname in files:
    #leitura da base
    data = pd.read_csv(fname, sep=',')
    data.dropna(inplace = True)
    data = data.to_numpy()
    
    statistic.kmeans(data, 3)

#Leitura da base
iris = datasets.load_iris()
names = iris.target_names
data = iris.data
labels = iris.target

clusters = statistic.kmeans(data, 3)
print(clusters)