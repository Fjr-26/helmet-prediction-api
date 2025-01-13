from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = "helmet_detect_model.h5"
IMG_SIZE = 50

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


def model_predict(image_file, model):
    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    pic_array = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if pic_array is None:
        raise ValueError("Unable to decode image file")

    new_array = cv2.resize(pic_array, (IMG_SIZE, IMG_SIZE))
    img_batch = np.expand_dims(new_array, axis=0)
    uimg = np.expand_dims(img_batch, axis=-1)

    uimg = uimg / 255.0

    prediction = model.predict(uimg)

    return prediction, pic_array.shape


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    allowed_extensions = {".jpg", ".jpeg", ".png"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        return (
            jsonify({"error": "Invalid file type. Allowed types: JPG, JPEG, PNG"}),
            400,
        )

    try:
        prediction, img_shape = model_predict(file, model)

        helmet_status = "with helmet" if prediction[0][0] <= 0.5 else "without helmet"

        response = {
            "prediction": float(prediction[0][0]),
            "helmet_status": helmet_status,
            "image_height": img_shape[0],
            "image_width": img_shape[1],
        }
        return jsonify(response), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
