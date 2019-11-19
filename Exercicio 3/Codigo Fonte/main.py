# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

"""
Created on Thu Nov 14 19:52:31 2019

@author: Murillo Freitas Bouzon

@project: PEL-208 Exercicio 3: Implementação do método LDA

"""

import statistic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


iris = datasets.load_iris()
data = iris.data
labels = iris.target
print(data)
print(labels)

novo = np.asarray([data[labels == i] for i in range(max(labels)+1)])
novo = novo.T