# Barcode Face Matching API

This simple Flask application provides an endpoint to match a captured face image against a mock database of known faces using the `face_recognition` library.

## Setup

1. Ensure you have Python 3.7+ installed.
2. Create a virtual environment and activate it:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
   > Note: `face_recognition` requires `dlib` which may need Visual Studio build tools on Windows.

## Running

```powershell
python app.py
```

The API will be available on `http://localhost:5000`.

### Endpoints

- `POST /api/match_face` – accepts JSON with an `image` field (face as base64); returns registration ID on successful face match. The server uses OpenCV's LBPH recognizer; you must pre‑populate `known_faces_db` with grayscale training images keyed by registration ID.
- `POST /api/scan` – accepts camera capture (base64). It first attempts to decode a barcode; if one is found it treats the decoded text as a registration ID lookup. If no barcode is present it falls back to the face recognizer.

Example JSON body for either endpoint:

```json
{ "image": "<base64 string>" }
```

### Enrolling faces

You can add members manually in `app.py` or via an admin route (not provided). For example:

```python
from app import known_faces_db
import cv2
img = cv2.imread('user123.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
known_faces_db['user123'] = gray
```

The recognizer is trained on this dictionary every time a request is made; for a production system you would persist a model and only retrain when new faces are added.



### Docker

A `Dockerfile` is included for containerizing the application. To build and run:

```powershell
# build image from project root
docker build -t barcode-face-matcher .

# run container, mapping port 5000
docker run --rm -p 5000:5000 barcode-face-matcher
```

The service will again listen on `http://localhost:5000` inside and outside the container.

## Example Request

```bash
curl -X POST http://localhost:5000/api/match_face \
  -H "Content-Type: application/json" \
  -d '{"image":"<base64 string>"}'
```

## Error Handling

The server responds with appropriate status codes for:

- missing or invalid JSON
- missing `image` field
- face not detected
- no match found

## Notes

This project uses a simple in-memory dictionary (`known_faces_db`). In production you should replace this with persistent storage.