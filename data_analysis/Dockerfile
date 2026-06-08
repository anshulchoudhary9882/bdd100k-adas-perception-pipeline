FROM python:3.12-slim

WORKDIR /app

# Install system graphics libraries required by OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the analysis scripts
COPY app.py .
COPY parser.py .

# Copy the pre-calculated reports folder into the container
COPY analysis_reports/ ./analysis_reports/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]