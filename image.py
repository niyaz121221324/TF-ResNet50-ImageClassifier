import numpy as np
from PIL import Image, UnidentifiedImageError

# Loads and preprocesses a single image.
def load_image(image_path, target_size):
    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize(target_size, Image.LANCZOS)
        return np.array(image) / 255.0
    except UnidentifiedImageError:
        print(f"UnidentifiedImageError: Unable to load {image_path}")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
    return None