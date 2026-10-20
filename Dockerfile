FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system libraries required by geospatial dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project configuration and source package
COPY pyproject.toml .
COPY src/ ./src/

# Install the project and its Python dependencies
RUN pip install --upgrade pip && \
    pip install .

# Copy the Streamlit application and trained models
COPY app.py .
COPY models/ ./models/

# Expose Streamlit's default port
EXPOSE 8501

# Start the application
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501"]