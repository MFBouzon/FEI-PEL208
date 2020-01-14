# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:17:46 2019

@author: Murillo

@project: PEL-208 Exercicio 7: Classificação de imagens utilizando Redes Neurais Convolucionais
"""

import numpy as np
from keras.applications import vgg16, vgg19, resnet50
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions


import keras
import keras.layers as layers
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

#Leitura da base MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = np.array(x_train) / 255
x_test = np.array(x_test) / 255

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

#instanciamento da rede
lenet = keras.Sequential()

#camadas convolucionais e maxpooling alternadas

lenet.add(keras.layers.Conv1D(
    filters=6,
    kernel_size=5,
    activation='relu',
    use_bias=True))
lenet.add(keras.layers.MaxPooling1D(pool_size=2, strides=2))

lenet.add(keras.layers.Conv1D(filters=16, kernel_size=5, activation='relu', use_bias=True))
lenet.add(keras.layers.MaxPooling1D(pool_size=2, strides=2))

lenet.add(keras.layers.Flatten())

#camadas MLP
lenet.add(layers.Dense(units=120, activation='relu'))

lenet.add(layers.Dense(units=84, activation='relu'))

lenet.add(layers.Dense(units=10, activation = 'softmax'))

lenet.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adadelta(),
    metrics=['accuracy']
)

#treinamento da rede
lenet.fit(x_train, y_train, epochs=15, batch_size=128)

#predição
y_pred = lenet.predict(x_test)

y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)


print(confusion_matrix(y_test, y_pred))
print(precision_recall_fscore_support(y_test, y_pred, average="micro"))

#definição das classes
classes = ["orange", "beer", "tiger"]

#Instancia o modelo VGG16
vgg_model = vgg16.VGG16(weights='imagenet')


#Instancia o modelo VGG19
vgg2_model = vgg19.VGG19(weights='imagenet')
 
#Instancia o modelo ResNet50
resnet_model = resnet50.ResNet50(weights='imagenet')
 
#para cada classe é lida uma das imagens de teste
for cl in classes:
    for i in range(1, 11):
        path = cl + str(i) +".jpg"
        img = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        #predição do modelo ResNet50
        preds = resnet_model.predict(x)
        print('ResNet Predicted:', decode_predictions(preds, top=1)[0])
        
        #predição do modeloo VGG16
        preds = vgg_model.predict(x)
        print('VGG16 Predicted:', decode_predictions(preds, top=1)[0])
        
        #predição do modelo VGG19
        preds = vgg2_model.predict(x)
        print('VGG19 Predicted:', decode_predictions(preds, top=1)[0])
