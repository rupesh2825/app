from flask import Flask, request, jsonify
import base64
import numpy as np
import io
from PIL import Image

# barcode / vision libraries
import cv2
from pyzbar import pyzbar

# we will use OpenCV's LBPH face recognizer (part of contrib package)


app = Flask(__name__)

# Mock Database: In production load face samples & barcode values from a persistent store.
# For barcode lookup we simply consider the decoded text as registrationId.  For faces
# we store a single grayscale image per member (or you could store a trained LBPH model).
# The matching routine below trains the LBPH recognizer on the fly for simplicity.
# Example to add a member manually:
#   img = cv2.imread('user123.jpg')
#   known_faces_db['user123'] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
known_faces_db = {}  # { registrationId: grayscale_image_array }

@app.route('/api/match_face', methods=['POST'])
def match_face():
    # ensure JSON payload
    if not request.is_json:
        return jsonify({"status": "invalid_request", "message": "Expected JSON body"}), 400

    data = request.get_json()
    captured_base64 = data.get('image')  # Base64 string from Android
    if not captured_base64:
        return jsonify({"status": "invalid_request", "message": "'image' field is required"}), 400

    try:
        # 1. Decode the captured image
        img_data = base64.b64decode(captured_base64)
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        img_np = np.array(img)
    except Exception as e:
        return jsonify({"status": "decode_error", "message": str(e)}), 400

    # convert to grayscale for recognition
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    # build recognizer if we have any enrolled faces
    if not known_faces_db:
        return jsonify({"status": "no_faces_registered"}), 404

    ids = []
    samples = []
    for idx, (rid, sample) in enumerate(known_faces_db.items()):
        ids.append(idx)
        samples.append(sample)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(samples, np.array(ids))

    label, confidence = recognizer.predict(gray)
    # lower confidence = better match; threshold may need tuning
    if confidence < 100:
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
        if code in known_faces_db:
            return jsonify({"status": "barcode_match", "registrationId": code})
        else:
            return jsonify({"status": "barcode_no_match", "barcode": code}), 404

    # no barcode -> try face recognition using LBPH
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    if not known_faces_db:
        return jsonify({"status": "no_faces_registered"}), 404

    ids = []
    samples = []
    for idx, (rid, sample) in enumerate(known_faces_db.items()):
        ids.append(idx)
        samples.append(sample)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(samples, np.array(ids))
    label, confidence = recognizer.predict(gray)
    if confidence < 100:
        matched_id = list(known_faces_db.keys())[label]
        return jsonify({"status": "match_found", "registrationId": matched_id})
    else:
        return jsonify({"status": "no_match"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)