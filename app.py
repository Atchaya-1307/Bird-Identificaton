from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from flask_cors import CORS  # Import CORS

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load the pre-trained model
model = load_model('BirdModel1.keras')

# Ensure these class names match the classes the model was trained with
# Replace with the actual class names your model was trained with
class_names = ['kingfisher', 'Painted Stork', 'Rose Ring Parakeet','Rock pigeon' ,'eagle' ,  'crow','peacock' , 'sparrow', 'woodpecker',]  # 9 classes in this case

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded file
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file provided'}), 400
        
        print("File received:", file.filename)
        
        # Open and process the image
        img = Image.open(file).convert('RGB')  # Ensure the image is in RGB format
        img = img.resize((128, 128))  # Resize to model input size
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Debugging: Check the shape of the image array
        print("Image shape:", img_array.shape)
        
        # Get predictions from the model
        predictions = model.predict(img_array)
        
        # Debugging: Print predictions
        print("Predictions:", predictions)
        
        # Get the predicted class index and its probability
        predicted_class_index = np.argmax(predictions[0])  # Index of the highest probability
        predicted_class = class_names[predicted_class_index]  # Class name based on index
        confidence = float(np.max(predictions[0]) * 100)  # Convert to float
        print(predicted_class)
        # Return the prediction and confidence as JSON
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence
        })
    except Exception as e:
        # Log the error
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
