# -*- coding: utf-8 -*-

"""
Created on Thu Oct 31 21:07:31 2019

@author: Murillo Freitas Bouzon

@project: PEL-208 Exercicio 2: Implementação da Análise de Componentes Principal (PCA)

"""

import statistic
import pandas as pd

data = pd.read_csv("alpswater.csv", sep=';', header = None)
data = data.values.tolist()

print(data)

print(statistic.getCovMatrix(data))