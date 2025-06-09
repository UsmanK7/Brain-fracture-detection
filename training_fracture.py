import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

def load_path(path, part):
    """
    Load X-ray dataset
    """
    dataset = []
    for folder in os.listdir(path):
        folder = os.path.join(path, folder)
        if os.path.isdir(folder):
            for body in os.listdir(folder):
                if body == part:
                    body_part = body
                    path_p = os.path.join(folder, body)
                    for id_p in os.listdir(path_p):
                        patient_id = id_p
                        path_id = os.path.join(path_p, id_p)
                        for lab in os.listdir(path_id):
                            if lab.split('_')[-1] == 'positive':
                                label = 'fractured'
                            elif lab.split('_')[-1] == 'negative':
                                label = 'normal'
                            path_l = os.path.join(path_id, lab)
                            for img in os.listdir(path_l):
                                img_path = os.path.join(path_l, img)
                                dataset.append(
                                    {
                                        'body_part': body_part,
                                        'patient_id': patient_id,
                                        'label': label,
                                        'image_path': img_path
                                    }
                                )
    return dataset


def trainPart(part):
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(THIS_FOLDER, 'Dataset')
    data = load_path(image_dir, part)
    labels = []
    filepaths = []

    for row in data:
        labels.append(row['label'])
        filepaths.append(row['image_path'])

    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    images = pd.concat([filepaths, labels], axis=1)

    train_df, test_df = train_test_split(images, train_size=0.9, shuffle=True, random_state=1)

    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,
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

    pretrained_model = tf.keras.applications.resnet50.ResNet50(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg')

    pretrained_model.trainable = False

    inputs = pretrained_model.input
    x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
    x = tf.keras.layers.Dense(50, activation='relu')(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(train_images, validation_data=val_images, epochs=1, callbacks=[callbacks])

    save_dir = os.path.join(THIS_FOLDER, "./plots/FractureDetection/", part)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    figAcc = plt.gcf()
    my_file = os.path.join(save_dir, "_Accuracy.jpeg")
    figAcc.savefig(my_file)
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    figAcc = plt.gcf()
    my_file = os.path.join(save_dir, "_Loss.jpeg")
    figAcc.savefig(my_file)
    plt.clf()

    model.save(os.path.join(THIS_FOLDER, "/weights/ResNet50_" + part + "_frac.h5"))

    results = model.evaluate(test_images, verbose=0)
    print(part + " Results:")
    print(results)
    print(f"Test Accuracy: {np.round(results[1] * 100, 2)}%")


categories_parts = ["Elbow", "Finger", "Forearm", "Hand", "Humerus", "Shoulder", "Wrist"]
for category in categories_parts:
    trainPart(category)
