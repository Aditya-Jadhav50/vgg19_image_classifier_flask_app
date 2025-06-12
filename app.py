from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model once at startup
model = load_model('vgg19.h5')

# Upload folder inside static for easy serving
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part in the request", 400
        f = request.files['file']
        if f.filename == '':
            return "No selected file", 400

        filename = secure_filename(f.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(file_path)

        preds = model_predict(file_path, model)
        decoded = decode_predictions(preds, top=1)
        result = decoded[0][0][1]  # Get predicted class label

        return render_template('index.html', filename=filename, prediction=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
