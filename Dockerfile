# Base image with Python
FROM python:3.13-slim

# set a work directory
WORKDIR /app

# copy requirements and install
COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake libopenblas-dev liblapack-dev libx11-6 libglib2.0-0 \
       libsm6 libxrender1 libfontconfig1 zlib1g-dev \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get remove -y build-essential cmake \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# copy application code
COPY . .

# expose port
EXPOSE 5000

# run the application
CMD ["python", "app.py"]
