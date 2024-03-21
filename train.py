# train.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Assuming your images are in a directory named 'data/train'
# and are divided into subdirectories 'positive' and 'negative'
train_datagen = ImageDataGenerator(rescale=1./255)  # rescale pixel values
train_generator = train_datagen.flow_from_directory(
        'data/train',  # directory where training data resides
        target_size=(32, 32),  # adjust based on your preprocessing
        batch_size=10,  # adjust based on your total number of images
        class_mode='binary')  # for two classes


# Simple model (Modify according to your needs)
model = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10)

# Save the model
model.save('model.h5')
