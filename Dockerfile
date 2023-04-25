# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

# Copy the backend code into the container at /app
COPY . /app/

# Install Node.js and npm
RUN apt-get install -y curl
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash -
RUN apt-get install -y nodejs

# Change the working directory to the frontend
WORKDIR /app/frontend

# Install the required packages for the frontend
RUN npm install

# Build the frontend
RUN npm run build

# Change the working directory back to /app
WORKDIR /app

# Expose the port the application will run on
EXPOSE 8000

# Start the Django development server and the React app
# CMD ["sh", "-c", "python manage.py makemigrations"]
# CMD ["sh", "-c", "python manage.py migrate"]
CMD ["sh", "-c", "python manage.py runserver 0.0.0.0:8000 & cd frontend && npm run start"]

