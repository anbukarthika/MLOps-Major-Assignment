import os
import joblib
import numpy as np
from flask import Flask, request, render_template
from PIL import Image

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'savedmodel.pth'
model = joblib.load(MODEL_PATH)

def preprocess_image(image_file):
    """Convert uploaded image to 64x64 grayscale and flatten to 4096 features."""
    img = Image.open(image_file).convert('L')   # grayscale
    img = img.resize((64, 64))
    img_array = np.array(img).reshape(1, -1)    # shape (1, 4096)
    # Normalize to [0,1] as in training data (original dataset already 0-1)
    img_array = img_array / 255.0
    return img_array

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            features = preprocess_image(file)
            prediction = model.predict(features)[0]
            return render_template('result.html', prediction=prediction)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)