FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and the rest of the application
COPY . .

# Install production runtime dependencies (all binary wheels) and local package
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.prod.txt \
    && pip install --no-cache-dir --no-deps -e .

# Expose the Flask port
EXPOSE 5000

# Set environment variables for production
ENV FLASK_APP=dashboard/app.py
ENV FLASK_ENV=production

# Run the application with Gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "dashboard.app:app"]
