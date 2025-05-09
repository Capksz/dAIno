# Base image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    chromium \ 
    chromium-driver \
    libgtk-3-dev \
    libnotify-dev \ 
    libgconf-2-4 \
    libnss3 \
    libxss1 \
    libasound2t64

# Set environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set workdir
WORKDIR /app

# Copy project files
COPY . /app/

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose Flask port
EXPOSE 5050

# Command to run Flask app
CMD ["python", "app.py"]
