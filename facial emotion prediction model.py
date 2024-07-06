import numpy as np
import cv2
import os
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# Define the path to the dataset
dataset_path = r'C:\Users\home\Desktop\ADT_Project\dataset'

# Define the emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Define the image dimensions
img_width, img_height = 48, 48

# Define the number of channels (grayscale)
num_channels = 1

# Define the number of classes (emotions)
num_classes = len(emotion_labels)

# Define the batch size and number of epochs
batch_size = 32
num_epochs = 60

# Define the data generators for the training, validation, and testing sets
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=os.path.join(dataset_path, 'train'),
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

test_generator = test_datagen.flow_from_directory(
    directory=os.path.join(dataset_path, 'test'),
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, num_channels)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model with an appropriate loss function, optimizer, and evaluation metric
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model on the training set and validate on the validation set
model.fit_generator(train_generator,
                    steps_per_epoch=train_generator.n // batch_size,
                    epochs=num_epochs)

# Evaluate the model on the testing set
scores = model.evaluate_generator(test_generator, steps=test_generator.n // batch_size)
print('Test accuracy:', scores[1])

model.save(r'C:\Users\home\Desktop\ADT_Project\model.h5')

# Use the trained model to predict emotions on new facial images provided by the user
img = cv2.imread(r'C:\Users\home\Desktop\1.jpg', 0)  # Read the image in grayscale
img = cv2.resize(img, (img_width, img_height))
img = np.expand_dims(img, axis=-1)
img = np.expand_dims(img, axis=0)
pred = model.predict(img)
emotion_label = emotion_labels[np.argmax(pred)]
print('Predicted emotion:', emotion_label)
