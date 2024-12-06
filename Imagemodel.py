import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Dataset Path
data_dir = 'E:\\Miniproject\\current Bird'  # Replace with your PNG dataset directory

# Load Dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    subset='training',
    validation_split=0.2,
    seed=123,
    image_size=(128, 128),
    batch_size=32
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    subset='validation',
    validation_split=0.2,
    seed=123,
    image_size=(128, 128),
    batch_size=32
)

# Prefetch for Performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_dataset.cache().shuffle(100).prefetch(buffer_size=AUTOTUNE)
val_ds = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# Normalization
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Data Augmentation
data_augmentation = Sequential([
    layers.RandomFlip("horizontal", input_shape=(128, 128, 3)),
    layers.RandomZoom(0.1),
    layers.RandomRotation(0.1),
])

# Model Definition
model = Sequential([
    data_augmentation,
    layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(train_dataset.class_names), activation='softmax')  # Adjust output for number of classes
])

# Compile the Model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Train the Model
history = model.fit(train_ds, validation_data=val_ds, epochs=10)

# Evaluate the Model
val_loss, val_accuracy = model.evaluate(val_ds)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Save the Model
model.save('BirdModel1.keras')
print("Model saved successfully!")

# Plot Loss and Accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.xlabel('Epoch')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.tight_layout()
plt.show()

# Prediction Function
def predict_bird(image_path, model_path='BirdModel.keras'):
    """
    Predict bird species from a PNG image.

    Args:
    - image_path (str): Path to the PNG image.
    - model_path (str): Path to the saved model.

    Returns:
    - Predicted class name.
    """
    # Load the Model
    model = load_model(model_path)
    print("Model loaded successfully!")

    # Load and Preprocess the PNG Image
    img = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
    img = img.resize((128, 128))  # Resize to model input size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(img_array)
    class_names = train_dataset.class_names  # Class names from training dataset
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]
    confidence = np.max(predictions[0]) * 100

    print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}%")

    # Display the Image
    plt.imshow(Image.open(image_path))
    plt.title(f"Predicted: {predicted_class} ({confidence:.2f}%)")
    plt.axis('off')
    plt.show()

    return predicted_class, confidence

# Example Usage
test_image_path = 'E:\Miniproject\current Bird\Kingfisher_bgremoved\kingfisher_3-removebg-preview.png'  # Replace with a test PNG image path
predict_bird(test_image_path)
