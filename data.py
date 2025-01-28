import os
import numpy as np
from image import load_image

# Loads and preprocesses image data and their labels from the specified directory.
def load_data(data_dir, target_size):
    images, labels = [], []
    class_indices = {
        class_name: idx for idx, class_name in enumerate(sorted(os.listdir(data_dir)))
        if os.path.isdir(os.path.join(data_dir, class_name))
    }

    for class_name, class_idx in class_indices.items():
        class_dir = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(class_dir, filename)
                image = load_image(image_path, target_size)
                if image is not None:
                    images.append(image)
                    labels.append(class_idx)
    return np.array(images), np.array(labels), class_indices

# Saves metadata about the model, such as class indices.
def save_metadata(model_folder, class_indices):
    metadata_path = os.path.join(model_folder, 'metadata.txt')
    with open(metadata_path, 'w') as metadata_file:
        metadata_file.write(str({'class_indices': class_indices}))