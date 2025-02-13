# Use official Python 3.11 image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy project files to the container
COPY . /app/

# Install dependencies
RUN pip install -r requirements.txt

# Expose port 8080 for the app
EXPOSE 8080

# Set entrypoint to run Streamlit app
ENTRYPOINT ["streamlit", "run", "Breast_Cancer_Analysis_App.py", "--server.port=8080", "--server.address=0.0.0.0"]
