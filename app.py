
import cv2
import face_recognition
import numpy as np
from flask import Flask, jsonify, request
from urllib.request import urlopen
from PIL import Image
from io import BytesIO

app = Flask(__name__)

def load_image_from_url(image_url):
    response = urlopen(image_url)
    image_bytes = BytesIO(response.read())
    image = Image.open(image_bytes)
    return np.array(image)

def detect_faces(image):
    face_locations = face_recognition.face_locations(image)
    return face_locations

def recognize_faces(image):
    face_encodings = face_recognition.face_encodings(image)

    # For simplicity, assume there's only one known face in this example
    known_face_encoding = known_face_encoding # Replace with the known face encoding

    results = {"total_faces": len(face_encodings), "known_faces": 0, "unknown_faces": 0}

    for face_encoding in face_encodings:
        # Compare each face found in the image with the known face
        match = face_recognition.compare_faces([known_face_encoding], face_encoding)

        if any(match):
            results["known_faces"] += 1
        else:
            results["unknown_faces"] += 1

    return results

@app.route("/check_faces", methods=["POST"])
def check_faces():
    try:
        data = request.get_json()
        image_url = data.get("image_url")

        if image_url:
            image = load_image_from_url(image_url)
            face_locations = detect_faces(image)
            recognition_results = recognize_faces(image)

            return jsonify({"face_locations": face_locations, "recognition_results": recognition_results})
        else:
            return jsonify({"error": "Image URL not provided"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)