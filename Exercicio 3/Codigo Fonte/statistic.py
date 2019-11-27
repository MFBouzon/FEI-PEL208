# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 19:57:10 2019

@author: Murillo

@project: PEL-208 Exercicio 3: Implementação do método LDA

Funções para o cálculo dos métodos estatísticos
"""

import numpy as np

#função para calcular a média de um conjunto de valores em uma lista
def mean(data):
    return sum(data)/len(data)

#função para calcular a média de cada característica de toda a amostra
def grand_mean(data):
    return np.asarray([mean(data[i]) for i in range(len(data))])

#método para calcular a matriz within-class
def within_class(data, classMeans):
    Sw = np.zeros((data.shape[1],data.shape[1]))
    for i in range(len(data)):
        Si = np.zeros((data.shape[1],data.shape[1]))
        for j in range(data.shape[2]):
            row, mv = data[i][:,j].reshape(data.shape[1], 1), classMeans[i].reshape(data.shape[1], 1)
            Si += (row - mv).dot((row-mv).T)
        Sw += Si
    return Sw

#método para calcular a matriz between-class
def between_class(grandMean, classMeans, data):
    Sb = np.zeros((data.shape[1], data.shape[1]))
    for i in range(len(classMeans)):
        meanV, meanG = classMeans[i].reshape(data.shape[1], 1), grandMean.reshape(data.shape[1], 1)    
        Sb += data[i].shape[1] * (meanV - meanG).dot((meanV - meanG).T)
    return Sb

#método que calcula a análise de discriminante linear para um conjunto de dados supervisionados
# utilizando a biblioteca numpy para o cálculo dos autovalores e autovetores
def LDA(data, labels):
    newData = np.asarray([data[:,labels == i] for i in range(max(labels)+1)])
    
    grandMean = grand_mean(data)
    grandMean.reshape(grandMean.shape[0], 1)
    
    classMeans = np.asarray([grand_mean(newData[i]) for i in range(len(newData))])
    
    Sw = within_class(newData, classMeans)
    Sb = between_class(grandMean, classMeans, newData)
    
    eigVal, eigVec = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    
    
    eigPair = [(np.abs(eigVal[i]), eigVec[:,i]) for i in range(len(eigVal))]
    eigPair = sorted(eigPair, key = lambda k: k[0], reverse = True)
    
    return eigPair

#método para transformar os dados originais para o novo
# eixo de coordenadas encontrados pelo método LDA
def LDA_transform(data, eigPair, k):
    W = []
    for i in range(k):
        W.append(eigPair[i][1].reshape(len(eigPair[i][1]), 1))
    W = np.asarray(W)
    data_lda = data.T.dot(W)
    return data_lda

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
 

#função que calcula a análise de componentes principais de um conjunto de dados utilizando
# o numpy para o cálculo dos auto valores e auto vetores
def PCA(data):
    covMatrix = getCovMatrix(data)
    eigenValues, eigenVector = np.linalg.eig(covMatrix)
    eigPair = [(np.abs(eigenValues[i]), eigenVector[:,i]) for i in range(len(eigenValues))]
    eigPair = sorted(eigPair, key = lambda k: k[0], reverse = True)
    
    return eigPair

#método para transformar os dados originais para o novo
# eixo de coordenadas encontrados pelo método PCA
def PCA_transform(data, eigPair, k):
    W = []
    for i in range(k):
        W.append(eigPair[i][1].reshape(len(eigPair[i][1]), 1))
    W = np.asarray(W)
    data_pca = data.T.dot(W)
    return data_pca

#método para transformar os trasnformados de volta para
# o eixo de coordenadas originais
def PCA_inverse_transform(data, eigVec, mean):
    mult = np.matmul(np.linalg.inv(eigVec.T), data)
    for i in range(len(mult)):
        for j in range(len(mult[i])):
            mult[i][j] + mean[i]
    return mult

    