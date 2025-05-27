FROM python:3.9-slim

# Εγκατάσταση git και άλλων system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Δημιουργία user για security
RUN useradd -m -u 1000 user

WORKDIR /code

# Αντιγραφή requirements και εγκατάσταση dependencies
COPY --chown=user requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Αντιγραφή υπόλοιπων αρχείων
COPY --chown=user . /code

USER user

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
