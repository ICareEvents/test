# Dockerfile
# Build a Docker image that runs your Streamlit or Flask app
# For Vercel, we'll read the PORT environment variable.

FROM python:3.10-slim

# Install system packages needed to build some Python libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3-distutils python3-dev \
 && rm -rf /var/lib/apt/lists/*

# Create a working dir
WORKDIR /app

# Copy your files into the container
COPY . /app

# Upgrade pip/setuptools/wheel
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# If it's a Streamlit app, you can run:
#   streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0
# If it's a Flask app, you can do:
#   gunicorn -b 0.0.0.0:$PORT app:app
# (where "app:app" is "filename:Flask-instance")

# For example, let's assume Streamlit:
EXPOSE 8080
ENV PORT=8080

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=$PORT", "--server.address=0.0.0.0"]
