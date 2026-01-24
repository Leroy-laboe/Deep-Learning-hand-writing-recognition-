# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
# Using tensorflow-cpu to keep the image smaller and avoid GPU overhead
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir tensorflow-cpu

# Copy the rest of the application code
COPY . .

# Expose the port that the app runs on
EXPOSE 7860

# Command to run the application
# Hugging Face Spaces expects the app to run on port 7860
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
