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
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = self._build_model()

    def _transform(self, x):   
        x = x / 255.
        x_resize = cv2.resize(x, (128,128))
        return x_resize

    def fit(self, img_loader):
        nb = len(img_loader)
        X = np.zeros((nb, 128, 128, 3))
        Y = np.zeros((nb,2))
        
        for i in range(nb):
            x, y = img_loader.load(i)
            X[i] = self._transform(x)
            Y[i, y] = 1
            
        self.model.fit(x=X, y=Y, batch_size=self.batch_size, epochs=self.epochs, verbose=1)

        
    def predict_proba(self, img_loader):
        nb = len(img_loader)
        X = np.zeros((nb, 128, 128, 3))
        for i in range(nb):
            X[i] = self._transform(img_loader.load(i))
        return self.model.predict(X)
    
    def _build_model(self):
        
        model = Sequential()
	
	# Initialize layers
        conv_layer1 = Conv2D(filters = 8, kernel_size = 5, activation='relu', input_shape=(360,640,3))
	conv_layer2 = Conv2D(filters = 16, kernel_size = 3, activation='relu')

	maxpool_layer = MaxPooling2D(pool_size=2)
	flatten_layer = Flatten()

	dense_layer1 = Dense(10,activation = 'relu')
	dense_layer2 = Dense(1,activation = 'sigmoid')

        # Add them to the model
	model.add(conv_layer1)
	model.add(conv_layer2)

	model.add(maxpool_layer)
	model.add(flatten_layer)

	model.add(dense_layer1)
	model.add(dense_layer2)

    	# Compile the model
        model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.01),
              metrics=['accuracy'])
        
        return model

