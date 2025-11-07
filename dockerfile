# Use official Python image as base
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Copy requirement file first (for caching)
COPY requirement.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Copy all project files
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app with uvicorn
CMD ["uvicorn", "scripts.api:app", "--host", "0.0.0.0", "--port", "8000"]
