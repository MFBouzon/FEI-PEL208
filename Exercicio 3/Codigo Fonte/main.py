# -*- coding: utf-8 -*-


"""
Created on Thu Nov 14 19:52:31 2019

@author: Murillo Freitas Bouzon

@project: PEL-208 Exercicio 3: Implementação do método LDA

"""

import statistic
import plot
from sklearn import datasets
from sklearn.decomposition import PCA

#Leitura da base
iris = datasets.load_iris()
names = iris.target_names
data = iris.data
labels = iris.target
data = data.T

#Aplicação do LDA
lda = statistic.LDA(data, labels)
#Aplicação do PCA
pca = statistic.PCA(data)

#Transformação dos dados utilizando dois autovetores
lda_data = statistic.LDA_transform(data, lda, 2)
pca_data = statistic.PCA_transform(data, pca, 2)

#Plotagem dos dados em duas dimensões
plot.transformedData2D(lda_data, labels, "LDA: Iris projetada nos dois primeiros discrimantes lineares", names)
plot.transformedData2D(pca_data, labels, "PCA: Iris projetada nas duas primeiras componentes principais", names)

#Transformação dos dados utilizando dois autovetores
lda_data = statistic.LDA_transform(data, lda, 1)
pca_data = statistic.PCA_transform(data, pca, 1)


#Plotagem dos dados em uma dimensão
plot.transformedData1D(lda_data, labels, "LDA: Iris projetada no primeiro discrimante linear", names)
plot.transformedData1D(pca_data, labels, "PCA: Iris projetada na primeira componente principal", names)


for i in range(1, 4):
    #pca_data = statistic.PCA_transform(data, pca, i)
    #pca_data = pca_data.reshape(pca_data.shape[1], pca_data.shape[0])
        
    pca2 = PCA(n_components = i)
    pca2.fit(data.T)
    pca_data = pca2.transform(data.T)

    lda = statistic.LDA(pca_data.T, labels)
    if i == 1:
        lda_data = statistic.LDA_transform(pca_data.T, lda, 1)    
        plot.transformedData1D(lda_data, labels, "LDA aplicado sobre " + str(i) + " componente(s) do PCA", names)
    else:
        lda_data = statistic.LDA_transform(pca_data.T, lda, 2)    
        plot.transformedData2D(lda_data, labels, "LDA aplicado sobre " + str(i) + " componente(s) do PCA", names)


