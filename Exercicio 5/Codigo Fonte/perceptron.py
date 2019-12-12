# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 00:24:27 2019

@author: Murillo

@project: PEL-208 Exercicio 5: Implementação de um Perceptron

Definição da classe Perceptron
"""

import numpy as np

class Perceptron():
    
    #Construtor da classe que recebe o número de dimensões "n", o número de iterações "epoch" e
    # a taxa de aprendizado "learning_rate"
    def __init__(self, n, epochs = 10, learning_rate=0.1):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.zeros(n + 1)
    
    #método para predizer a classe dada uma entrada de tamanho "n"
    def predict(self, inputs):
        total_sum = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if total_sum > 0:
            return 1
        else:
            return 0
        
    #método para realizar o treinamento do perceptron recebendo os dados de treinamento e 
    # os seus respectivos labels
    def train(self, data_train, labels):
        for i in range(self.epochs):
            for inp, lab in zip(data_train, labels):
                prediction = self.predict(inp)
                self.weights[1:] += self.learning_rate*(lab - prediction) * inp
                self.weights[0] += self.learning_rate*(lab - prediction)
                