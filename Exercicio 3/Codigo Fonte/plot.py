# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 19:03:56 2019

@author: Murillo Freitas Bouzon

@project: PEL-208 Exercicio 3: Implementação do método LDA

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

#Função para plotar o gráfico com os dados projetados nos dois primeiros eigenVectors
def transformedData2D(transData, labels, title, names):
    for label, marker, color in zip(range(3), ('>', "^", 'v'), ('blue', 'red', 'green')):
        plt.scatter(x=transData[:,0].real[labels == label], 
                    y = transData[:,1].real[labels == label],
                    marker = marker,
                    color = color,
                    alpha = 0.8,
                    label = names[label])
            
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title(title)
    
    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom=False, top=False,  
            labelbottom=True, left=False, right=False, labelleft=True)
    
    plt.grid()
    plt.tight_layout
    plt.show()


#Função para plotar o gráfico com os dados projetados no primeiro eigen-vector
def transformedData1D(transData, labels, title, names):
    for label, marker, color in zip(range(3), ('>', "^", 'v'), ('blue', 'red', 'green')):
        plt.scatter(x=transData[:,0].real[labels == label], 
                    y = labels[labels == label],
                    marker = marker,
                    color = color,
                    alpha = 0.8,
                    label = names[label])
            
    plt.xlabel("PC1")
    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title(title)
    
    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom=False, top=False,  
            labelbottom=True, left=False, right=False, labelleft=True)
    
    plt.grid()
    plt.tight_layout
    plt.show()