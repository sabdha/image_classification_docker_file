import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def build_model(n_classes):
    base_model = Xception(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    head_model = base_model.output
    head_model = Flatten()(head_model)
    head_model = Dense(512, activation='relu')(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(n_classes, activation='softmax')(head_model)

    model = Model(inputs=base_model.input, outputs=head_model)

    for layer in base_model.layers:
        layer.trainable = False

    return model

def data_pipeline(batch_size, train_data_path, valid_path, eval_path):
    train_augment = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    valid_augment = ImageDataGenerator(rescale=1./255)
    eval_augment = ImageDataGenerator(rescale=1./255)

    train_gen = train_augment.flow_from_directory(
        train_data_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    valid_gen = valid_augment.flow_from_directory(
        valid_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    eval_gen = eval_augment.flow_from_directory(
        eval_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, valid_gen, eval_gen

def number_of_images(directory):
    count = 0
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext.lower() in ['.png', '.jpg', '.jpeg']:
                count += 1
    return count

def trainer(data_path, batch_size, epochs):
    path_train_data = os.path.join(data_path, 'train')
    path_valid_data = os.path.join(data_path, 'validation')
    path_eval_data = os.path.join(data_path, 'test')

    total_train_img = number_of_images(path_train_data)
    total_valid_img = number_of_images(path_valid_data)
    total_eval_img = number_of_images(path_eval_data)

    train_gen, valid_gen, eval_gen = data_pipeline(batch_size, path_train_data, path_valid_data, path_eval_data)

    n_classes = len(train_gen.class_indices)

    model = build_model(n_classes)
    optimizer = Adam(lr=1e-5)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.fit(
        train_gen,
        steps_per_epoch=total_train_img // batch_size,
        epochs=epochs,
        validation_data=valid_gen,
        validation_steps=total_valid_img // batch_size
    )

    print("Training is completed")
    print("[INFO] Evaluation phase...")

    predictions = model.predict(eval_gen)
    prediction_index = np.argmax(predictions, axis=1)

    my_classification_report = classification_report(eval_gen.classes, prediction_index, target_names=eval_gen.class_indices.keys())
    my_conf_matrix = confusion_matrix(eval_gen.classes, prediction_index)

    print('[INFO] Classification Report: ')
    print(my_classification_report)

    print('[INFO] Confusion Matrix: ')
    print(my_conf_matrix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="Batch size", default=32)
    parser.add_argument("--epochs", type=int, help="Training epochs", default=10)
    args = parser.parse_args()

    data_path = './dummy_data'  # Replace with your dataset directory
    trainer(data_path, args.batch_size, args.epochs)
