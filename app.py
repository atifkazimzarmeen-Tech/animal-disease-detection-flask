from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from tensorflow.keras.applications.resnet50 import preprocess_input

app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = tf.keras.models.load_model("final_resnet50_animal_disease.keras")

# Class labels (EXACT training order)
class_names = [
    "cat_cat_ringworm",
    "cat_cat_scabies",
    "cat_dermatitis",
    "cat_fine",
    "cat_flea_allergy",
    "dog_Dog_Ringworm",
    "dog_Dog_Scabies",
    "dog_Healthy_Dog",
    "fish_Aeromoniasis_Bacterial_diseases",
    "fish_Bacterial_Red_disease",
    "fish_Bacterial_disease_gill",
    "fish_Fungal_Saprolegniasis_diseases",
    "fish_Healthy_Fish",
    "fish_Parasitic_diseases",
    "fish_Viral_White_disease_tail"
]

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_url = None

    if request.method == "POST":
        file = request.files["image"]

        if file and file.filename != "":
            # Save image
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)

            # Open image
            image = Image.open(image_path).convert("RGB")
            processed_image = preprocess_image(image)

            preds = model.predict(processed_image)
            class_index = int(np.argmax(preds))

            prediction = class_names[class_index]
            confidence = round(float(np.max(preds)) * 100, 2)

            image_url = image_path  # for HTML display

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_url=image_url
    )

if __name__ == "__main__":
    app.run(debug=True)
