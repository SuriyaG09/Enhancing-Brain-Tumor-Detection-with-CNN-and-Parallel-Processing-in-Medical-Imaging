import time
start_time = time.time()
import cv2 as cv
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Activation, Dropout, Flatten, Dense, MaxPooling2D

path = "datasets/"
no_tumor_images = os.listdir(path+'no/')
yes_tumor_images = os.listdir(path+'yes/')

dataset = []
label = []
INPUT_SIZE = 64

#print(no_tumor_images)

for i, image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=="jpg"):
        image = cv.imread(path+'no/'+image_name) 
        
        image = Image.fromarray(image,'RGB')
        image = image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=="jpg"):
        image = cv.imread(path+'yes/'+image_name) 
        
        image = Image.fromarray(image,'RGB')
        image = image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)
x_train, x_test, y_train, y_test = train_test_split(dataset,label, test_size = 0.2, random_state = 0)

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = (INPUT_SIZE, INPUT_SIZE,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics= ['accuracy'])

model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=10, validation_data=(x_test,y_test), shuffle=False)
print("model created")
model.save("Braintumor_model_sequential1.h5")
end_time = time.time()
total_time = end_time - start_time
print("Total time taken:", total_time)