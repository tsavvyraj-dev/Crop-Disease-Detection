from flask import Flask, render_template, request, redirect, send_from_directory, url_for
import numpy as np
import json
import uuid
import os
import tensorflow as tf

# ---------------- APP SETUP ----------------
app = Flask(__name__)

# Reduce TensorFlow log noise (important on cloud)
tf.get_logger().setLevel('ERROR')

# Base directory (IMPORTANT for deployment)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- PATH CONFIG ----------------
MODEL_PATH = os.path.join(BASE_DIR, "models", "plant_disease_recog_model_pwp.keras")
JSON_PATH = os.path.join(BASE_DIR, "plant_disease.json")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploadimages")

# Create upload folder automatically (cloud-safe)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model(MODEL_PATH)

# ---------------- LABELS ----------------
label = [
 'Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Background_without_leaves',
 'Blueberry___healthy',
 'Cherry___Powdery_mildew',
 'Cherry___healthy',
 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn___Common_rust',
 'Corn___Northern_Leaf_Blight',
 'Corn___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy'
]

# ---------------- LOAD JSON ----------------
with open(JSON_PATH, 'r') as file:
    plant_disease = json.load(file)

# ---------------- ROUTES ----------------

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


# ---------------- IMAGE PROCESSING ----------------
def extract_features(image_path):
    image = tf.keras.utils.load_img(image_path, target_size=(160, 160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.array([feature])
    return feature


def model_predict(image_path):
    img = extract_features(image_path)
    prediction = model.predict(img)
    prediction_label = plant_disease[prediction.argmax()]
    return prediction_label


# ---------------- UPLOAD HANDLER ----------------
@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    if request.method == "POST":
        image = request.files['img']

        unique_name = f"temp_{uuid.uuid4().hex}_{image.filename}"
        save_path = os.path.join(UPLOAD_FOLDER, unique_name)

        image.save(save_path)

        prediction = model_predict(save_path)

        return render_template(
            'home.html',
            result=True,
            imagepath=f'/uploadimages/{unique_name}',
            prediction=prediction
        )

    return redirect('/')


# ---------------- LOCAL RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
