# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 15:44:34 2019

@author: Murillo

@project: PEL-208 Exercicio 4: Implementação do algoritmo de clusterização K-Means

Métodos estatísticos
"""

import numpy as np
import plot

#método para cálculo de distância
def dist(x1, x2, r):
    return sum(abs(x1 - x2)**r)**(1/r)

#função para calcular a média de um conjunto de valores em uma lista
def mean(data):
    return sum(data)/len(data)

#função para calcular a média de cada característica de toda a amostra
def grand_mean(data):
    return np.asarray([mean(data[i]) for i in range(len(data))])

#método para clusterizar dados utilizando o kmeans
def kmeans(data, k):
    C = np.asarray([np.random.uniform(np.min(data[:,i]), np.max(data[:,i]), size=k) 
    for i in range(data.shape[1])]).T
    
    olderC = np.zeros(C.shape)
    clusters = np.zeros(len(data))
    error = sum(dist(C, olderC, 2))
    
    while error != 0:
        
        for i in range(len(data)):
            distance = [dist(data[i], C[j], 2) for j in range(len(C))]
            clusters[i] = np.argmin(distance)
        
        olderC = C.copy()
        
        for i in range(k):
            points = np.asarray([data[j] for j in range(len(data)) if clusters[j] == i])
            if(len(points) > 0):
                C[i] = grand_mean(points.T)
            
        if(data.shape[1] == 2):
            plot.plotClusters(data, clusters, C)
        
        error = sum(dist(C, olderC, 2))
        
    return clusters
