
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets,layers,models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

#loading dataset from mnist
(train_images,train_labels),(test_images,test_labels)=datasets.mnist.load_data()
#print(np.array(train_images[0]))
# plt.title(train_labels[0])
# plt.imshow(train_images[0])
# plt.show()

#preprocessing : turn image data into 0,1
train_images=train_images/255.0
test_images=test_images/255.0
#print(np.array(train_images[0]))
# plt.title(train_labels[0])
# plt.imshow(train_images[0])
# plt.show()

#turn data into 28 by 28 pixle and color value of 1 it means grayscale
train_images=train_images.reshape((train_images.shape[0],28,28,1)) 
test_images=test_images.reshape((test_images.shape[0],28,28,1))
#print(np.array(train_images[0]))
#print(train_labels[0])
# convert lebels into one hot encoded formate
train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)
#print(np.argmax(train_labels[0]))

# Build Sequential Models with CNN layers
model = models.Sequential()

#First Convolutional layer
model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))

#Second Convolutional layer
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

# Third Convolutional layer
model.add(layers.Conv2D(64,(3,3), activation='relu'))

#Flatten from 3D output to 1D and add dense layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

#output layer with 10 neurons (10 digit classes)
model.add(layers.Dense(10,activation='softmax'))

# Compile the model 
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images,train_labels,epochs=5,batch_size=64,validation_data=(test_images,test_labels))

model.save('models/mnist.keras')


test_loss,test_acc=model.evaluate(test_images,test_labels)

print(f"Test Accuracy is : {test_acc*100:.2f}%")

#make predictions
predictions=model.predict(test_images)
print(f"Prediction for test image : {np.argmax(predictions[0])}")
plt.title(f"Prediction Label : {predictions[0].argmax()}")
plt.imshow(test_images[0].reshape(28,28),cmap='gray')
plt.show()


#load_models=tf.keras.models.load_model('models/mnist')