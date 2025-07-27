# Use slim Python base with amd64 compatibility
FROM --platform=linux/amd64 python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code, PDFs, collections, etc.
COPY . .

# Copy the pre-downloaded SentenceTransformer model (you MUST download it locally first!)
# COPY model ./model

# Set Python to unbuffered mode for proper logging
ENV PYTHONUNBUFFERED=1

# Run your main script
ENTRYPOINT ["python", "main.py"]
