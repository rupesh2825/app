# Base image with Python
FROM python:3.13-slim

# set a work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libzbar0 \
    && rm -rf /var/lib/apt/lists/*

# copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy application code
COPY . .

# copy application code
COPY . .

# expose port
EXPOSE 5000

# run the application
CMD ["python", "app.py"]
