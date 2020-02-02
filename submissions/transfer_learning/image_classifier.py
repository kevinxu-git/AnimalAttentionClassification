import numpy as np
import cv2
import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras import applications

class ImageClassifier(object):

    def __init__(self, batch_size=64, epochs=2):
        """The constructor. Defines some training hyperparameters."""
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = self._build_model()

    def _transform(self, x):   
        """Transforms the image so it has a shape of (128, 128, 3) and values between 0 and 1"""
        """Any other kind of transformation should be put here."""
        x = x / 255.
        x_resize = cv2.resize(x, (128,128))
        return x_resize

    def fit(self, X_train, y_train):
        """Fits the model"""
        n = len(X_train)
        X = np.zeros((n, 128, 128, 3))
        Y = y_train.copy()
        for i in range(n):
            X[i] = self._transform(X_train[i])
            
        self.model.fit(x=X, y=Y, batch_size=self.batch_size, epochs=self.epochs, verbose=1)

        
    def predict(self, X_test):
        """Performs the prediction."""
        n = len(X_test)
        X = np.zeros((nb, 128, 128, 3))
        for i in range(nb):
            X[i] = self._transform(X_test[i])
        return self.model.predict(X)
    
    def _build_model(self):
        base_model = applications.VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(128,128,3)),classes=2)

        for layer in base_model.layers:  
            layer.trainable = False
        
        x = base_model.output  
        x = Flatten()(x)  
        x = Dense(10, activation='elu')(x)
        x = Dropout(0.4)(x)  
        x = Dense(10, activation='elu')(x)
        x = Dropout(0.1)(x)
        predictions = Dense(1, activation='sigmoid', name='predictions')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=1e-3), metrics=['accuracy'])
        
        return model