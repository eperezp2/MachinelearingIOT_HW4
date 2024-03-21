import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = load_model('model.h5')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'data/train',  # Adjust this path to where your test data is located
    target_size=(32, 32),  # Should match the input size of your model
    batch_size=10,  # Adjust based on your test set size
    class_mode='binary')  # Or 'categorical' if you have more than two classes

loss, accuracy = model.evaluate(test_generator)

print(f'Test accuracy: {accuracy}')
