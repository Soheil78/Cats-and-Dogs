# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 13:48:14 2024

@author: sh032
"""

import numpy as np
import os
import tensorflow
import keras
from keras import layers
from keras import Sequential
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#Importation of pictures
def generate(path,max_images_per_class=1000):
    images = []
    labels = []
    for classe in os.listdir(path):
        chemin_classe = os.path.join(path, classe)
        if os.path.isdir(chemin_classe):
            
            image_count=0
            
            for fichier in os.listdir(chemin_classe):
                
                if image_count >= max_images_per_class:
                    break
                
                chemin_image = os.path.join(chemin_classe, fichier)
                if os.path.isfile(chemin_image):
                    try:
                        img = Image.open(chemin_image)
                        img = img.resize((300, 300))
                        img = np.array(img)
                        images.append(img)
                        labels.append(classe)
                        
                        image_count += 1
                        
                    except Exception as e:
                        print(f"Erreur lors du traitement de l'image {chemin_image}: {e}")
                        
    return images, labels


path=r"C:\Users\sh032\PetImages"

images,labels=generate(path)

for i in range(len(images)):
    if np.ndim(images[i])==2:
        images[i]=np.stack([images[i]] * 3, axis=-1)


images2=[]
labels2=[]


for i in range(len(images)):
    if images[i].shape==(300,300,3):
        images2.append(images[i])
        labels2.append(labels[i])

images2=np.array(images2)


transformer=LabelEncoder()
labels2=transformer.fit_transform(labels2)


labels2=np.array(labels2).reshape((len(labels2),1))

#Split of the dataset

X_train,X_test=train_test_split(images2, test_size=0.2)
y_train,y_test=train_test_split(labels2, test_size=0.2)

#Normalization of data

X_train=X_train/255
X_test=X_test/255

#The model

model=keras.Sequential([
    keras.Input((300,300,3)),
    
    layers.Conv2D(32, 3, padding="valid"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPool2D(pool_size=(2,2),strides=2),
    
    layers.Conv2D(64, 3, padding="valid"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPool2D(pool_size=(2,2),strides=2),
    
    layers.Conv2D(128, 3, padding="valid"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPool2D(pool_size=(2,2),strides=2),
    
    layers.Flatten(),
    layers.Dense(2,activation="softmax")
       
])

model.summary()

model.compile(   
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    metrics=["accuracy"]    
    )

model.fit(X_train,y_train,epochs=5,verbose=2)

#To make new predictions

def prediction(picture_path):
  pic=Image.open(picture_path)
  pic=pic.resize(300,300)
  pic=np.array(pic)
  pic=np.expand_dims(picture,axis=0)
  return np.argmax(model.predict(pic))

