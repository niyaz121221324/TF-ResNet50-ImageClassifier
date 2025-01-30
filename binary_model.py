import datetime
import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from preprocess import resolve_folder_path, configure_gpu

def build_binary_model(freeze_base=True, learning_rate=0.001):
    """
    Builds and compiles a binary classification model based on ResNet50V2.

    Args:
        freeze_base (bool): Whether to freeze the base model layers during training. Default is True.
        learning_rate (float): Learning rate for the Adam optimizer. Default is 0.001.

    Returns:
        tensorflow.keras.models.Model: The compiled binary classification model.
    """
    # Load the ResNet50V2 base model with pre-trained ImageNet weights
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Add custom classification layers
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)  # Binary classification
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(model, train_gen, val_gen, batch_size, epochs):
    return model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // batch_size,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=val_gen.samples // batch_size
    )

def create_binary_model():
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
        dataset_path, target_size=image_size, batch_size=batch_size, class_mode='binary', subset='training'
    )
    val_gen = datagen.flow_from_directory(
        dataset_path, target_size=image_size, batch_size=batch_size, class_mode='binary', subset='validation'
    )
    
    model = build_binary_model()
    history = train_model(model, train_gen, val_gen, batch_size, epochs=10)
    
    model_folder = os.path.join(save_path, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(model_folder, exist_ok=True)
    model.save(os.path.join(model_folder, 'binary_model.h5'))
    
    print(f"Binary model saved at: {model_folder}")