# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose ports
EXPOSE 8000
EXPOSE 8501

# Default command (we override in docker-compose)
CMD ["uvicorn","api_app:app","--host","0.0.0.0","--port","8000"]
