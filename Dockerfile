FROM python:3.11-slim

# Install library sistem yang diperlukan
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    python3-dev \
    pkg-config \
    && apt-get clean

# Set direktori kerja
WORKDIR /app

# Salin semua file proyek ke container
COPY . .

# Install dependensi Python
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Jalankan aplikasi
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
