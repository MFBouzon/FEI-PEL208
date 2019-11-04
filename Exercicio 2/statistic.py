# -*- coding: utf-8 -*-

"""
Created on Thu Oct 31 21:07:31 2019

@author: Murillo Freitas Bouzon

@project: PEL-208 Exercicio 2: Implementação da Análise de Componentes Principal (PCA)

"""

def mean(data):
    return sum(data)/len(data)

def covariance(x, y):
    
    a = [i - mean(x) for i in x]
    b = [i - mean(y) for i in y]
 
    c = [a[i]*b[i] for i in range(len(a))]
    
    return sum(c)/(len(x)-1)
 
def getCovMatrix(data):
    cov = [[covariance(data[x],data[y]) for x in range(len(data))] for y in range(len(data))]
    return cov