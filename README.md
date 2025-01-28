# TF-ResNet50-ImageClassifier

Этот проект представляет собой приложение для создания и настройки классификатора изображений на основе модели ResNet50V2, используя TensorFlow и Keras. Вы можете как создать новую модель, так и дообучить существующую модель с использованием предоставленного набора данных.

## Основные функции

- **Создание новой модели**: Автоматическая настройка и обучение классификатора изображений на базе ResNet50V2.
- **Дообучение существующей модели**: Возможность дообучить уже существующую модель с использованием нового набора данных.
- **Использование GPU**: Проект поддерживает использование GPU для ускорения обучения (если доступно).
- **Аугментация данных**: Поддерживается увеличение данных с помощью ImageDataGenerator (вращение, сдвиги, отражение и т. д.).
- **Поддержка произвольного числа классов**: Модель автоматически адаптируется к числу классов в наборе данных.

## Структура проекта

- `configure_gpu`: Настраивает TensorFlow для использования GPU (если доступно).
- `load_image`: Загружает и обрабатывает отдельное изображение.
- `load_data`: Загружает данные из указанного каталога, предварительно обрабатывая изображения и метки классов.
- `build_model`: Создает архитектуру классификатора на базе ResNet50V2.
- `train_model`: Обучает модель на основе предоставленных данных.
- `create_new_model`: Функция для создания и обучения новой модели.
- `fine_tune_model`: Функция для дообучения существующей модели.
- `save_metadata`: Сохраняет метаданные, включая индексы классов.
- `main`: Основное меню для взаимодействия с приложением.

## Использование

1. Убедитесь, что у вас установлен Python 3.8+ и необходимые зависимости:
   ```bash
   pip install -r requirements.txt
   ```
2. Запуск приложения
   ```bash
   python main.py
   ```
Данный проект на выходе предоставляет файл формата h5

## Пример использования
```python
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

# Загрузка модели
model = load_model('model_path.h5')

# Загрузка изображения
img_path = 'path_to_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
image = image.convert("RGB")
image = image.resize(target_size, Image.LANCZOS)
image_array = np.array(imag) / 255.0

# Предсказание
predictions = model.predict(img_array)
print("Class probabilities:", predictions)
```
