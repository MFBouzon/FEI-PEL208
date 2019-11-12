# -*- coding: utf-8 -*-

"""
Created on Thu Oct 31 21:07:31 2019

@author: Murillo Freitas Bouzon

@project: PEL-208 Exercicio 2: Implementação da Análise de Componentes Principal (PCA)

"""
from math import sqrt
import numpy as np

#função para calcular a média de um conjunto de valores em uma lista
def mean(data):
    return sum(data)/len(data)

#função para calcular a covariância entre duas variáveis
def covariance(x, y):
    
    a = [i - mean(x) for i in x]
    b = [i - mean(y) for i in y]
 
    c = [a[i]*b[i] for i in range(len(a))]
    
    return sum(c)/(len(x)-1)
 
#função que retorna a matriz de covariância entre os dados da matriz    
def getCovMatrix(data):
    cov = [[covariance(data[x],data[y]) for x in range(len(data))] for y in range(len(data))]
    return cov
 
#funão para resolver um sistema linear de duas equações        
def solveLin(mat, v):
    y = v - mat[0][0]
    x = mat[0][1]
    S = sqrt((x**2) + (y**2))
    return x/S, y/S

#função que retorna os auto valores de uma matriz
def getEigenValues(data):
    
    T = data[0][0] + data[1][1]
    D = (data[0][0]*data[1][1]) - (data[0][1]*data[1][0])    

    L1 = T/2 + sqrt(T**2/4 - D)
    L2 = T/2 - sqrt(T**2/4 - D)
    
    return L1, L2

#função que retorna os auto vetores a partir de uma matriz e seus auto valores
def getEigenVector(cov, eValues):
    eVec = []
    for i in eValues:
        res = solveLin(cov, i)
        eVec.append(res)
    return eVec

#função que calcula a análise de componentes principais de um conjunto de dados utilizando
# o numpy para o cálculo dos auto valores e auto vetores
def PCA(data):
    covMatrix = getCovMatrix(data)
    eigenValues, eigenVector = np.linalg.eig(covMatrix)
    zipped = zip(eigenValues, eigenVector.T)
    zipped = sorted(zipped)
    
    eigVec = []
    eigVal = []
    
    for i in zipped:
        eigVal.append(i[0])
        eigVec.append(i[1])
   
    eigVal = eigVal[::-1]
    eigVec = eigVec[::-1]
    eigVec = np.asarray(eigVec)
    
    return eigVal, eigVec

#função que calcula as componentes principais utilizando a implementação própria
# do cálculo dos auto valores e auto vetores para 2 variáveis
def PCA2(data):
    covMatrix = getCovMatrix(data)
    eigenValues = getEigenValues(covMatrix)
    eigenVector = getEigenVector(covMatrix, eigenValues)
    return eigenValues, eigenVector
    