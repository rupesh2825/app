from flask import Flask, request, jsonify
import face_recognition
import base64
import numpy as np
import io
from PIL import Image

app = Flask(__name__)

# Mock Database: In production, load these encodings from your SQL/NoSQL DB
# format: { "registrationId": numpy_encoding_array }
# Example (populate at startup or via separate admin API):
# known_faces_db['user123'] = face_recognition.face_encodings(face_recognition.load_image_file('user123.jpg'))[0]
known_faces_db = {} 

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
        img = Image.open(io.BytesIO(img_data))
        img_np = np.array(img)
    except Exception as e:
        return jsonify({"status": "decode_error", "message": str(e)}), 400

    # 2. Find encoding for the captured face
    captured_encodings = face_recognition.face_encodings(img_np)
    
    if not captured_encodings:
        return jsonify({"status": "no_face_detected"}), 400

    captured_encoding = captured_encodings[0]

    # 3. Compare with database
    for reg_id, known_encoding in known_faces_db.items():
        matches = face_recognition.compare_faces([known_encoding], captured_encoding, tolerance=0.6)
        if matches[0]:
            return jsonify({
                "status": "match_found",
                "registrationId": reg_id
            })

    return jsonify({"status": "no_match"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)