from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random
mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

model=keras.models.load_model('models/mnist.keras')
model.summary()
print("================================\n\n\n\n\n")
prediction=model.predict(test_images)
print(f"{random.randrange(0,12000)}")
test_index=random.randrange(0,12000)
print(f"Prediction {np.argmax(prediction[test_index])}")
plt.title(f"Prediction of this : {np.argmax(prediction[test_index])}")
plt.imshow(test_images[test_index].reshape(28,28),cmap='gray')
plt.show()

