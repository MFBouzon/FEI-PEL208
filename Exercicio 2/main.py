# -*- coding: utf-8 -*-

"""
Created on Thu Oct 31 21:07:31 2019

@author: Murillo Freitas Bouzon

@project: PEL-208 Exercicio 2: Implementação da Análise de Componentes Principal (PCA)

"""

import statistic
import plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

#bases
files = ["alpswater.csv", "US Census.csv", "Books_attend_grade.csv"]

#calcular o PCA para cada base
for fname in files:
    #leitura da base
    data = pd.read_csv(fname, sep=';', header = None)
    data = data.values.T.tolist()
    #Aplicação do PCA
    eigenValues, eigenVector = statistic.PCA(data)
    print('Eigen Values = ', eigenValues,"\nEigen Vector = ", eigenVector)
    #Transformação dos dados para os novos eixos
    transData = statistic.PCA_transform(data, eigenVector)
    
    #Transformação inversa dos dados para os eixos originais
    mean = [np.mean(i) for i in data]  
    invData = statistic.PCA_inverse_transform(transData, eigenVector, mean)
    
    #Gráfico com a taxa de quanto cada eixo explica sobre os dados
    plot.explainedVariance(eigenValues) 
    #Gráfico com os dados apresentados nos novos eixos
    plot.transformedData(transData, fname[:len(fname)-4])
    
    #Comparação do PCA com a Regressão Linear
    inf = [min(data[i]) for i in range(len(data)-1)]
    sup = [max(data[i]) for i in range(len(data)-1)]
    
    inp = np.asarray(data[0:(len(data)-1)]).T
    out = np.array([data[len(data)-1]]).T
    reg = LinearRegression()
    reg.fit(inp,out)
    
    
    x = [np.linspace(inf[i], sup[i], len(data[i])) for i in range(len(data)-1)]
    x = np.asmatrix(x).T
    yl = reg.predict(x)
    
    pca = PCA(n_components = 1)
    
    xy_pca = pca.fit_transform(np.transpose(data))
    xy_n = pca.inverse_transform(xy_pca)
   
    plot.PCAxRegression(inp, out, xy_n, x, yl) 
    

    