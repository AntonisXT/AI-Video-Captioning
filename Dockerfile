FROM python:3.9-slim

WORKDIR /code

# Install git and other system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt /code/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all application files
COPY . /code

# Expose port 8501
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.headless", "true", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false", "--server.fileWatcherType", "none"]
