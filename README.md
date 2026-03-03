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