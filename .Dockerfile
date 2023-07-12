# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Generate JWT secret
RUN python -c "import secrets; print(secrets.token_urlsafe())" > jwt_secret.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV JWT_SECRET=$(cat jwt_secret.txt)

# Run gunicorn when the container launches
CMD ["gunicorn", "-w", "4", "server.__main__:app"]
