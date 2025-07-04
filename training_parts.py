# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 01:02:52 2024

@author: fkk42
"""

import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


# load images to build and train the model
#                       ....                                     /    img1.jpg
#             test      Hand            patient0000   positive  --   img2.png
#           /                /                         \    .....
#   Dataset   -         Elbow  ------   patient0001
#           \ train               \         /                           img1.png
#                       Shoulder        patient0002     negative --      img2.jpg
#                       ....                   \
#
from PIL import Image

def load_path(path):
    """
    Load X-ray dataset
    """
    dataset = []
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            for body in os.listdir(folder_path):
                body_path = os.path.join(folder_path, body)
                for id_p in os.listdir(body_path):
                    patient_id = id_p
                    id_path = os.path.join(body_path, patient_id)
                    for lab in os.listdir(id_path):
                        if lab.endswith('positive'):
                            label = 'fractured'
                        elif lab.endswith('negative'):
                            label = 'normal'
                        lab_path = os.path.join(id_path, lab)
                        for img in os.listdir(lab_path):
                            img_path = os.path.join(lab_path, img)
                            try:
                                # Attempt to open the image file
                                image = Image.open(img_path)
                                # If successful, append to the dataset
                                dataset.append({'label': body, 'image_path': img_path})
                            except Exception as e:
                                # If an error occurs (e.g., corrupted file), skip the image
                                print(f"Error loading image: {img_path}. Skipping...")
                                continue
    return dataset



# load data from path
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
image_dir = THIS_FOLDER + '/Dataset'
data = load_path(image_dir)
labels = []
filepaths = []

# add labels for dataframe for each category 0-Elbow, 1-Hand, 2-Shoulder
Labels = ["Elbow", "Finger","Forearm","Hand", "Humerus", "Shoulder","Wrist"]
for row in data:
    labels.append(row['label'])
    filepaths.append(row['image_path'])

filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

images = pd.concat([filepaths, labels], axis=1)

# split all dataset 10% test, 90% train (after that the 90% train will split to 20% validation and 80% train
train_df, test_df = train_test_split(images, train_size=0.9, shuffle=True, random_state=1)

# each generator to process and convert the filepaths into image arrays,
# and the labels into one-hot encoded labels.
# The resulting generators can then be used to train and evaluate a deep learning model.

# now we have 10% test, 72% training and 18% validation

train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
    validation_split=0.2)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=64,
    shuffle=True,
    seed=42,
    subset='training'
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=64,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

# we use rgb 3 channels and 224x224 pixels images, use feature extracting , and average pooling
pretrained_model = tf.keras.applications.resnet50.ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg')

# for faster performance
pretrained_model.trainable = False

inputs = pretrained_model.input
x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
x = tf.keras.layers.Dense(50, activation='relu')(x)
outputs = tf.keras.layers.Dense(len(Labels), activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)
print(model.summary())

# Adam optimizer with low learning rate for better accuracy
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# early stop when our model is over fit or vanishing gradient, with restore best values
callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(train_images, validation_data=val_images, epochs=2,
                    callbacks=[callbacks])

# save model to this path
model.save(THIS_FOLDER + "/weights/ResNet50_BodyParts.h5")
results = model.evaluate(test_images, verbose=0)
print(results)
print(f"Test Accuracy: {np.round(results[1] * 100, 2)}%")


# create plots for accuracy and save it
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# create plots for loss and save it
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()