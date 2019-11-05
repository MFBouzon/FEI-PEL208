# -*- coding: utf-8 -*-

"""
Created on Thu Oct 31 21:07:31 2019

@author: Murillo Freitas Bouzon

@project: PEL-208 Exercicio 2: Implementação da Análise de Componentes Principal (PCA)

"""

import statistic
import pandas as pd

data = pd.read_csv("data.csv", sep=';', header = None)
data = data.values.T.tolist()
print(data)
covMatrix = statistic.getCovMatrix(data)
print(covMatrix)
eigenValues = statistic.getEigenValues(covMatrix)
eigenVector = statistic.getEigenVector(covMatrix, eigenValues)

print(eigenValues, "\n", eigenVector)

data2 = [[2, 1], [1,2]]
print(statistic.getEigenVector(data2, statistic.getEigenValues(data2)))