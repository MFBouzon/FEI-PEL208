# -*- coding: utf-8 -*-

"""
Created on Thu Oct 31 21:07:31 2019

@author: Murillo Freitas Bouzon

@project: PEL-208 Exercicio 2: Implementação da Análise de Componentes Principal (PCA)

"""

import statistic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

#bases
files = ["alpswater.csv", "US Census.csv", "data.csv"]

#calcular o PCA para cada base
for fname in files:
    data = pd.read_csv(fname, sep=';', header = None)
    data = data.values.T.tolist()
    eigenValues, eigenVector = statistic.PCA(data)
    
    print('Eigen Values = ', eigenValues,"\nEigen Vector = ", eigenVector)
    
    
    inf = min(data[0])
    sup = max(data[0])
    
    
    inp = np.asarray([data[0]]).T
    out = np.array([data[1]]).T
    reg = LinearRegression().fit(inp,out)
    
    x = np.linspace(inf, sup, len(data[0]))
    y = eigenVector[0][0]*x + eigenVector[0][1] 
    z = eigenVector[1][0]*x + eigenVector[1][1]
    
    yl = reg.predict(np.asmatrix(x).T)
    
    
    plt.plot(data[0], data[1],'o', color='blue')
    plt.plot(x, y, '-r', label= 'PC 1')
    plt.plot(x, z, '-g', label= 'PC 2')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('PCA')
    plt.plot(np.asmatrix(x).T, yl, color = 'black', label = 'Regressão Linear')
    plt.legend(loc='upper left')
    plt.show()
    
    data2 = [(data[0][i]*eigenVector[0][0]) + (data[1][i]*eigenVector[1][0]) for i in range(len(data[0]))]
    data3 = [(data[0][i]*eigenVector[0][1]) + (data[1][i]*eigenVector[1][1]) for i in range(len(data[0]))]
    plt.plot(data2, data3, 'o', color='red')
    
    plt.show()
    
    pca = PCA(n_components = 2)
    pca.fit(np.transpose(data))
    components = pca.components_
    
    
    data2 = [(data[0][i]*components[0][0]) + (data[1][i]*components[1][0]) for i in range(len(data[0]))]
    data3 = [(data[0][i]*components[0][1]) + (data[1][i]*components[1][1]) for i in range(len(data[0]))]
    plt.plot(data2, data3, 'o', color='red')
    plt.show()
    
    plt.plot(components)
    plt.plot(eigenVector)
    plt.show()   
    print(eigenVector)
    print(components)
    print(pca.explained_variance_)
    