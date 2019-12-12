# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 01:11:25 2019

@author: Murillo


@project: PEL-208 Exercicio 5: Implementação de um Perceptron

Funções para plotagem
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def perceptron_result(data, labels, title, names=["",""]):
    colors = ['r', 'b', 'g']
    markers = ['x', 'o', '^']
    
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    for i in range(len(data)):
        plt.scatter(data[i][0], data[i][1], s=100, c=colors[labels[i]], marker=markers[labels[i]])
    
    patch1 = mpatches.Patch(color='red', label=names[0])
    patch2 = mpatches.Patch(color='blue', label=names[1])
    plt.legend(handles=[patch1, patch2])
    plt.show()