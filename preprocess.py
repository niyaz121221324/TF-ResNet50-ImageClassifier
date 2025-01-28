import os
import tensorflow as tf

def resolve_folder_path(target_folder_name: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # определяем полный путь
    full_path = os.path.join(current_dir, target_folder_name)

    if os.path.exists(full_path):
        return full_path
    else:
        raise FileNotFoundError(f"The path {full_path} does not exist.")

# Configures TensorFlow to use the GPU if available.
def configure_gpu():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"TensorFlow is using GPU: {physical_devices[0]}")
    else:
        print("No GPU detected. TensorFlow will use CPU.")