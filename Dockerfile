# Χρήση Python 3.11 για συμβατότητα με τις dependencies σου
FROM python:3.11-slim

# Εγκατάσταση system dependencies για OpenCV και άλλες libraries
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Δημιουργία user για security
RUN useradd -m -u 1000 user

# Ορισμός working directory
WORKDIR /app

# Αντιγραφή requirements και εγκατάσταση dependencies
COPY --chown=user ./requirements.txt requirements.txt

# Εγκατάσταση Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Αντιγραφή όλων των αρχείων
COPY --chown=user . /app

# Δημιουργία απαραίτητων directories
RUN mkdir -p data/videos/app_demo results temp && \
    chown -R user:user /app

# Αλλαγή στον user
USER user

# Environment variables
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONWARNINGS=ignore::FutureWarning
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health

# Εκτέλεση της εφαρμογής
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
