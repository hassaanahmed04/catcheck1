from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


model_path = "D:/shayaan/script/catcheck/model/keras_Model.h5" 
model = load_model(model_path, compile=False)


class_names_path = "D:/shayaan/script/catcheck/model/labels.txt"  
with open(class_names_path, "r") as f:
    class_names = [line.strip() for line in f]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    try:
        if 'file' not in request.files:
            raise ValueError('No file part')

        file = request.files['file']

        if file.filename == '':
            raise ValueError('No selected file')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            file.save(file_path)

            img = Image.open(file_path).convert("RGB")
            size = (224, 224)
            img = img.resize(size, Image.LANCZOS)

            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            normalized_img_array = (img_array.astype(np.float32) / 127.5) - 1

            predictions = model.predict(normalized_img_array)
            index = np.argmax(predictions)
            numeric_label = str(index)  # Extract the numeric label
            class_name = class_names[index]
            confidence_score = predictions[0][index]

            result_str = f"The detected class is: {class_name[2:]} with confidence {confidence_score:.2f}"

            return jsonify({'result': result_str, 'numeric_label': numeric_label, 'class': class_name[2:], 'confidence': float(confidence_score), 'image_path': os.path.abspath(file_path)}), 200, {'Content-Type': 'application/json'}

        else:
            raise ValueError('File type not allowed')

    except Exception as e:
        return jsonify({'error': str(e), 'class': 'Unknown Class', 'confidence': 0})



if __name__ == '__main__':
    app.run(debug=True)
