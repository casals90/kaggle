#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 16:55:33 2018

@author: jordicasals
"""

import keras
import pandas as pd

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

from sklearn.neighbors import KNeighborsClassifier


#################
### Constants ###
#################

TRAINIG_DATA = "data/train.csv"
TEST_DATA = "data/test.csv"

BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 45

############
### Main ###
############
    
if __name__ == "__main__":
    
    ###############################
    ### Traing data (train.csv) ###
    ##############################
    
    df_train = pd.read_csv(TRAINIG_DATA,
                           header=None,
                           skiprows=[0],
                           low_memory=False)
    
    # Get labels from column 1
    y_train = df_train[0]
    # Select all colums without 0 (labels)
    X_train = df_train.loc[:, 1:]
    
    X_train = X_train.astype('float32')
    X_train = normalize(X_train, norm='l2')
    
    print(X_train.shape[0], 'train samples')
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    
    # Split the dataset in train and test
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, 
                                                                y_train, 
                                                                test_size=0.2, 
                                                                random_state=90)
    
    X_test = pd.read_csv(TEST_DATA,
                         header=None,
                         skiprows=[0])
    
    ######################
    ### Neural network ###
    ######################
    
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    model.summary()
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    
    history = model.fit(X_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=1,
                        validation_data=(X_validate, y_validate))
    
    score = model.evaluate(X_validate, y_validate, verbose=0)
    
    print('#'*10)
    print('Deep Learning:')
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    ##########################
    ### Predict (test.csv) ###
    ##########################
    
    y_pred = model.predict_classes(X_test)
    
    data = {'ImageId': [i for i in range(1, X_test.shape[0] + 1)],
            'Label': y_pred}
    
    df_prediction = pd.DataFrame(data)
    
    df_prediction.to_csv("submissions/digit_recognizer_eval_nn.csv",
    				       header=True,
                         sep=',',
                         index=False,
                         encoding='utf-8')
    
    ###########
    ### KNN ###
    ###########
    
    clf_knn = KNeighborsClassifier(n_neighbors=NUM_CLASSES)
    clf_knn.fit(X_train, y_train)
    
    print('#'*10)
    print('Knn:')
    print("Test accuracy: ", clf_knn.score(X_validate, y_validate))
    
    y_pred = clf_knn.predict(X_test)
    
    data = {'ImageId': [i for i in range(1, X_test.shape[0] + 1)],
            'Label': y_pred}
    
    df_prediction_knn = pd.DataFrame(data)
    
    df_prediction_knn.to_csv("submissions/digit_recognizer_eval_knn.csv",
    				           header=True,
                             sep=',',
                             index=False,
                             encoding='utf-8')
    