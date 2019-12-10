# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 15:31:05 2019

@author: Murillo

@project: PEL-208 Exercicio 4: Implementação do algoritmo de clusterização K-Means

Funções para exibição de gráficos

"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#método para plotar os clusters e seus centroides
def plotClusters(data, clusters, centroids):
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.style.use('ggplot')
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    
    for i in range(len(centroids)):
        plt.scatter(data[clusters == i,0], data[clusters == i,1], s=20, c=colors[i], label="cluster "+str(i+1))
       
    plt.scatter(centroids[:,0], centroids[:,1], marker='o', s=200, c='black')     
    plt.legend()
    plt.show()     