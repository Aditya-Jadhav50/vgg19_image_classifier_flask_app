# vgg19_image_classifier_flask_app# VGG19 Image Classifier Flask App

This project is a Flask web application that performs image classification using a pretrained VGG19 model. Users upload images through a simple web interface, and the app predicts the top ImageNet class labels for the uploaded image.

---

## Project Overview

The app leverages the Keras implementation of the VGG19 convolutional neural network pretrained on ImageNet. The Flask backend handles image upload, preprocessing (resizing, normalization), and model inference. The prediction results are displayed alongside the uploaded image.

---

## Folder Structure

- `app.py` — Main Flask application handling routes and prediction logic.  
- `vgg19_model.py` — Script to load and save the VGG19 model weights to `vgg19.h5`.  
- `vgg19.h5` — Saved Keras VGG19 model file.  
- `requirements.txt` — Python dependencies for the project.  
- `/templates/` — HTML templates: `base.html` (base layout) and `index.html` (upload & results page).  
- `/static/uploads/` — Folder to store uploaded images for display.

---

## Setup Instructions

1. Create and activate a Python virtual environment.  
2. Install dependencies:  
```bash
pip install -r requirements.txt

##Notes
Uploaded images are saved in /static/uploads/ for rendering on the results page.

Model predictions utilize Keras’ decode_predictions to map output probabilities to human-readable labels.

Image preprocessing conforms to VGG19 input requirements (224x224 resizing, normalization).