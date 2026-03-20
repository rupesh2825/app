from flask import Flask, request, jsonify
import base64
import numpy as np
import io
from PIL import Image
import os

# barcode / vision libraries
import cv2
from pyzbar import pyzbar

# Check for face recognition module
try:
    cv2.face
except AttributeError:
    raise ImportError("opencv-contrib-python is required for face recognition")

# we will use OpenCV's LBPH face recognizer (part of contrib package)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
FACE_SIZE = (200, 200)
CONFIDENCE_THRESHOLD = 100  # Tune this based on your images


app = Flask(__name__)

# Mock Database: In production load face samples & barcode values from a persistent store.
# For barcode lookup we simply consider the decoded text as registrationId.  For faces
# we store a single grayscale image per member (or you could store a trained LBPH model).
# The matching routine below trains the LBPH recognizer on the fly for simplicity.
# Example to add a member manually:
#   img = cv2.imread('user123.jpg')
#   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#   faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#   if len(faces) > 0:
#       x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
#       face_roi = gray[y:y+h, x:x+w]
#       known_faces_db['user123'] = cv2.resize(face_roi, FACE_SIZE)
#       known_registrations.add('user123')
known_faces_db = {}  # { registrationId: preprocessed_face_array }
known_registrations = set()  # separate store for barcode registrations

def load_registrations():
    # Load barcode registrations from a file
    reg_file = 'registrations.txt'
    if os.path.exists(reg_file):
        with open(reg_file, 'r') as f:
            for line in f:
                known_registrations.add(line.strip())

def load_known_faces():
    # Load and preprocess face samples
    faces_dir = 'known_faces'
    if os.path.exists(faces_dir):
        for filename in os.listdir(faces_dir):
            if filename.endswith(('.jpg', '.png')):
                rid = os.path.splitext(filename)[0]
                img = cv2.imread(os.path.join(faces_dir, filename))
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    if len(faces) > 0:
                        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                        face_roi = gray[y:y+h, x:x+w]
                        known_faces_db[rid] = cv2.resize(face_roi, FACE_SIZE)

load_registrations()
load_known_faces()

@app.route('/api/match_face', methods=['POST'])
def match_face():
    # ensure JSON payload
    if not request.is_json:
        return jsonify({"status": "invalid_request", "message": "Expected JSON body"}), 400

    data = request.get_json()
    captured_base64 = data.get('image')  # Base64 string from Android
    if not captured_base64:
        return jsonify({"status": "invalid_request", "message": "'image' field is required"}), 400

    # Handle data URL
    if "," in captured_base64:
        captured_base64 = captured_base64.split(",", 1)[1]

    try:
        # 1. Decode the captured image
        img_data = base64.b64decode(captured_base64)
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        img_np = np.array(img)
    except Exception as e:
        return jsonify({"status": "decode_error", "message": str(e)}), 400

    # convert to grayscale for recognition
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return jsonify({"status": "no_face_detected"}), 400

    # take the largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_roi = gray[y:y+h, x:x+w]
    face_img = cv2.resize(face_roi, FACE_SIZE)

    # build recognizer if we have any enrolled faces
    if not known_faces_db:
        return jsonify({"status": "no_faces_registered"}), 404

    ids = []
    samples = []
    for idx, (rid, sample) in enumerate(known_faces_db.items()):
        ids.append(idx)
        samples.append(sample)  # already preprocessed and resized
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(samples, np.array(ids))

    label, confidence = recognizer.predict(face_img)
    # lower confidence = better match; threshold may need tuning
    if confidence < CONFIDENCE_THRESHOLD:
        matched_id = list(known_faces_db.keys())[label]
        return jsonify({"status": "match_found", "registrationId": matched_id})
    else:
        return jsonify({"status": "no_match"}), 404

@app.route('/api/scan', methods=['POST'])
def scan_image():
    """Try barcode first; if none found, fall back to face matching.
    Expects JSON with 'image' containing base64-encoded camera capture."""
    if not request.is_json:
        return jsonify({"status": "invalid_request", "message": "Expected JSON body"}), 400

    data = request.get_json()
    img_b64 = data.get('image')
    if not img_b64:
        return jsonify({"status": "invalid_request", "message": "'image' field is required"}), 400

    # Handle data URL
    if "," in img_b64:
        img_b64 = img_b64.split(",", 1)[1]

    try:
        img_data = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        img_np = np.array(img)
    except Exception as e:
        return jsonify({"status": "decode_error", "message": str(e)}), 400

    # attempt barcode decode
    barcodes = pyzbar.decode(img_np)
    if barcodes:
        # pick first barcode
        code = barcodes[0].data.decode('utf-8')
        # treat the code as registrationId lookup
        if code in known_registrations:
            return jsonify({"status": "barcode_match", "registrationId": code})
        else:
            return jsonify({"status": "barcode_no_match", "barcode": code}), 404

    # no barcode -> try face recognition using LBPH
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    # detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return jsonify({"status": "no_face_detected"}), 400

    # take the largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_roi = gray[y:y+h, x:x+w]
    face_img = cv2.resize(face_roi, FACE_SIZE)
    if not known_faces_db:
        return jsonify({"status": "no_faces_registered"}), 404

    ids = []
    samples = []
    for idx, (rid, sample) in enumerate(known_faces_db.items()):
        ids.append(idx)
        samples.append(sample)  # already preprocessed and resized
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(samples, np.array(ids))
    label, confidence = recognizer.predict(face_img)
    if confidence < CONFIDENCE_THRESHOLD:
        matched_id = list(known_faces_db.keys())[label]
        return jsonify({"status": "match_found", "registrationId": matched_id})
    else:
        return jsonify({"status": "no_match"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)