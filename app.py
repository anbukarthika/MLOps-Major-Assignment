import joblib
import numpy as np
from flask import Flask, request, render_template
from PIL import Image
import os

app = Flask(__name__)
model = joblib.load('savedmodel.pth')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file):
    img = Image.open(file).convert('L')
    img = img.resize((64, 64))
    img_array = np.array(img).reshape(1, -1)
    img_array = img_array.astype(np.float32) / 255.0
    return img_array

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file uploaded", 400
        file = request.files['image']
        if file.filename == '':
            return "Empty filename", 400
        if not allowed_file(file.filename):
            return "Unsupported image format. Please upload PNG, JPG, or JPEG.", 400
        try:
            features = preprocess_image(file)
            prediction = model.predict(features)[0]
            return render_template('result.html', prediction=int(prediction))
        except Exception as e:
            return f"Error processing image: {str(e)}", 500
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)