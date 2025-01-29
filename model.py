import datetime
import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from preprocess import resolve_folder_path, configure_gpu
from data import load_data, save_metadata
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def build_model(num_classes, freeze_base=True, learning_rate=0.001):
    """
    Builds and compiles a ResNet50V2-based model for classification.

    Args:
        num_classes (int): Number of classes for the output layer.
        freeze_base (bool): Whether to freeze the base model layers during training. Default is True.
        learning_rate (float): Learning rate for the Adam optimizer. Default is 0.001.

    Returns:
        tensorflow.keras.models.Model: The compiled classification model.
    """
    # Validate inputs
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer.")

    # Load the ResNet50V2 base model with pre-trained ImageNet weights
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Add custom classification layers
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Trains the model using the provided data generators.
def train_model(model, train_gen, val_gen, batch_size, epochs):
    return model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // batch_size,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=val_gen.samples // batch_size
    )

# Creates and trains a new classification model.
def create_new_model():
    configure_gpu()

    save_path = resolve_folder_path(input("Enter the folder to save the new model: "))
    if not os.path.exists(save_path):
        print("Invalid folder path.")
        return

    dataset_path = resolve_folder_path(input("Enter the dataset folder path: "))
    batch_size, image_size = 32, (224, 224)

    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        dataset_path, target_size=image_size, batch_size=batch_size, class_mode='categorical', subset='training'
    )
    val_gen = datagen.flow_from_directory(
        dataset_path, target_size=image_size, batch_size=batch_size, class_mode='categorical', subset='validation'
    )

    model = build_model(len(train_gen.class_indices))
    history = train_model(model, train_gen, val_gen, batch_size, epochs=10)

    model_folder = os.path.join(save_path, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(model_folder, exist_ok=True)
    model.save(os.path.join(model_folder, 'model.keras'))

    save_metadata(model_folder, train_gen.class_indices)
    print(f"Model saved at: {model_folder}")

# Fine-tunes an existing model using a new dataset.
def fine_tune_model():
    configure_gpu()
    save_path = resolve_folder_path(input("Enter the folder to save the fine-tuned model: "))
    if not os.path.exists(save_path):
        print("Invalid folder path.")
        return

    model_path = resolve_folder_path(input("Enter the path to the existing model: "))
    if not os.path.exists(model_path):
        print("Model file does not exist.")
        return

    dataset_path = resolve_folder_path(input("Enter the new dataset folder path: "))
    batch_size, image_size = 32, (224, 224)

    images, labels, class_indices = load_data(dataset_path, target_size=image_size)
    if not class_indices:
        print("No data found for fine-tuning.")
        return

    labels_one_hot = to_categorical(LabelEncoder().fit_transform(labels), num_classes=len(class_indices))

    model = tf.keras.models.load_model(model_path)
    for layer in model.layers:
        layer.trainable = True

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        images, labels_one_hot, batch_size=batch_size, epochs=10
    )

    model_folder = os.path.join(save_path, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(model_folder, exist_ok=True)
    model.save(os.path.join(model_folder, 'fine_tuned_model.h5'))

    save_metadata(model_folder, class_indices)
    print(f"Fine-tuned model saved at: {model_folder}")