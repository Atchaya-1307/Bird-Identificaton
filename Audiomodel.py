import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
import soundfile as sf

# Dataset Path      
data_dir = 'E:\Miniproject\Birds Audio'  # Update with your dataset path

# Helper Function to Extract MFCC Features
def preprocess_audio(file_path, n_mfcc=13, max_pad_len=128):
    """
    Preprocess an audio file to extract MFCC features.
    """
    try:
        audio, sample_rate = librosa.load(file_path, sr=22050, mono=True)  # Downsample
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        pad_width = max_pad_len - mfcc.shape[1]
        if pad_width > 0:
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        return mfcc
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# Parallel Processing Helper Function
def process_file(args):
    """
    Helper function for parallel processing.
    """
    file_path, label_index, max_pad_len = args
    mfcc = preprocess_audio(file_path, max_pad_len=max_pad_len)
    if mfcc is not None:
        return mfcc, label_index
    return None

# Load Dataset Using Parallel Processing
def load_audio_dataset(directory, labels, max_pad_len=128):
    """
    Load audio dataset, extract MFCC features, and return features and targets.
    """
    args = []
    for label in labels:
        class_dir = os.path.join(directory, label)
        for file in os.listdir(class_dir):
            if file.endswith('.wav'):
                file_path = os.path.join(class_dir, file)
                args.append((file_path, labels.index(label), max_pad_len))
    
    features, targets = [], []
    with Pool() as pool:
        results = list(tqdm(pool.imap(process_file, args), total=len(args), desc="Processing Dataset"))
    for result in results:
        if result:
            features.append(result[0])
            targets.append(result[1])
    return np.array(features), np.array(targets)

# Inspect Dataset Files
def check_audio_files(directory, labels):
    """
    Verify that all audio files are valid and readable.
    """
    for label in labels:
        class_dir = os.path.join(directory, label)
        for file in os.listdir(class_dir):
            if file.endswith('.wav'):
                try:
                    with sf.SoundFile(os.path.join(class_dir, file)) as f:
                        pass
                except Exception as e:
                    print(f"Invalid file: {file}, Error: {e}")

# Define Classes (Folders in Dataset Directory)
labels = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

# Check Audio Files
check_audio_files(data_dir, labels)

# Load and Split Data
features, targets = load_audio_dataset(data_dir, labels)
features = features[..., np.newaxis]  # Add a channel dimension for CNN
dataset = tf.data.Dataset.from_tensor_slices((features, targets))
train_size = int(0.8 * len(dataset))
train_ds = dataset.take(train_size).shuffle(100).batch(32)
val_ds = dataset.skip(train_size).batch(32)

# Model Definition
model = Sequential([
    layers.Input(shape=(13, 128, 1)),  # Shape based on MFCC output
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(labels), activation='softmax')
])

# Compile the Model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Train the Model
history = model.fit(train_ds, validation_data=val_ds, epochs=10)

# Evaluate the Model
val_loss, val_accuracy = model.evaluate(val_ds)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Save the Model
model.save('BirdAudioModel.keras')
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
def predict_bird_audio(file_path, model_path='BirdAudioModel.keras'):
    """
    Predict bird species from an audio file.

    Args:
    - file_path (str): Path to the audio file.
    - model_path (str): Path to the saved model.

    Returns:
    - Predicted class name and confidence.
    """
    model = load_model(model_path)
    print("Model loaded successfully!")
    mfcc = preprocess_audio(file_path)
    if mfcc is not None:
        mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension
        mfcc = mfcc[..., np.newaxis]  # Add channel dimension
        predictions = model.predict(mfcc)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = labels[predicted_class_index]
        confidence = np.max(predictions[0]) * 100
        print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}%")
        return predicted_class, confidence
    else:
        print("Error processing the audio file.")
        return None, None

# Example Usage
test_audio_path = 'E:\Miniproject\Birds Audio\peacock\XC191122 - Indian Peafowl - Pavo cristatus.mp3'  # Replace with a test audio file path
predict_bird_audio(test_audio_path)