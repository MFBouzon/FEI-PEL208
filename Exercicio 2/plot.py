# -*- coding: utf-8 -*-
"""
Created on Wed Nov 1 19:03:56 2019

@author: Murillo Freitas Bouzon

@project: PEL-208 Exercicio 2: Implementação da Análise de Componentes Principal (PCA)

Funções para exibição de gráficos

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Função para plotar o gráfico com o total da variância explicada por cada auto vetor
def explainedVariance(eigenValues):
    left = [i for i in range(len(eigenValues))]
    tick_label = ["PC" + str(i+1) for i in range(len(eigenValues))]
    right = [i*100 / sum(eigenValues) for i in eigenValues]
    rect = plt.bar(left, right, tick_label = tick_label, width = 0.5, label = right)
    plt.yticks(np.arange(0, 100+1, step=20))
    for r in rect:
        height = r.get_height()
        plt.text(r.get_x() + r.get_width()/2.0, height, '%f %%' % float(height), ha='center', va='bottom')
    plt.title("Variância Explicada")
    plt.show()

#Função para plotar o gráfico com os dados projetados nas duas componentes principais    
def transformedData(transData, title):
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.ylim(min(transData[1]) - abs(min(transData[1])*0.1), max(transData[1]) + abs(max(transData[1])*0.1))
    plt.plot(transData[0], transData[1], 'o', color='red')
    plt.show()

#função para plotar o gráfico com a reta encontrada pela Regressão Linear e a primeira componente principal
def PCAxRegression(inp, out, xy_n, x, yl):
    if(inp.shape[1] < 2):
        plt.plot(inp, out, 'o', color='black')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('PCA')
        plt.plot(x, yl, color = 'blue', label = 'Regressão Linear')
        plt.plot(xy_n[:,0], xy_n[:,1], color='red', label = 'Componente Principal')
        plt.legend(loc='upper left')
        plt.show()
    else:
        ax = plt.axes(projection="3d")
        ax.set_title("PCA")
        ax.scatter3D(inp[:,0], inp[:,1], out, color='black')
        ax.plot3D(xy_n[:,0], xy_n[:,1], xy_n[:,2], 'red')
        
        ax.plot3D(np.array(x[:,0]).ravel(), np.array(x[:,1]).ravel(), yl.flatten(), 'blue')
        plt.show()
        