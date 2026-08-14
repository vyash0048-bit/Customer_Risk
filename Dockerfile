FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required for psycopg2 and building packages
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and the rest of the application
COPY . .

# Install dependencies using the lightweight production requirements
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.prod.txt

# Expose the Flask port
EXPOSE 5000

# Set environment variables for production
ENV FLASK_APP=dashboard/app.py
ENV FLASK_ENV=production

# Run the application with Gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "dashboard.app:app"]
