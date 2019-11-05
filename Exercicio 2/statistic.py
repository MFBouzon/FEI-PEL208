# -*- coding: utf-8 -*-

"""
Created on Thu Oct 31 21:07:31 2019

@author: Murillo Freitas Bouzon

@project: PEL-208 Exercicio 2: Implementação da Análise de Componentes Principal (PCA)

"""
from math import sqrt
import numpy as np

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

def bhaskara(a, b, c):
    
    delta = b**2 - (4*a*c)
    if delta < 0:
        return []
    elif delta == 0:
        return [(-b)/(2*a)]
    else:
         return [((-b)+sqrt(delta))/(2*a), ((-b)-sqrt(delta))/(2*a)]

def gauss (a, b):
  n = len(a)
  for k in range(0, n-1, 1):
    for i in range(k+1, n, 1):
      l = a[i][k] / a[k][k]
      for j in range(k, n, 1):
        a[i][j] = a[i][j] - l * a[k][j]
      b[i] = b[i] - l * b[k]
  return(b, a)
    
def solveLin(mat, v):
    n = len(mat)
    vec = np.empty(n)
    vec.fill(1)
    for i in range(len(mat)):
        mat[i][i] = mat[i][i] - v
    if np.linalg.det(mat) != 0:  
        ret = np.dot(vec, np.linalg.inv(mat))
    return ret

def getEigenValues(data):
    
    a = 1
    b = -(data[0][0]+data[1][1])
    c = (data[0][0]*data[1][1]) - (data[1][0]*data[0][1])
    
    print(a, b, c)
    
    return bhaskara(a, b, c)

def getEigenVector(cov, eValues):
    eVec = []
    for i in eValues:
        res = solveLin(cov, i)
        eVec.append(res.tolist())
    return eVec