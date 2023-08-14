# Use a slim version of Python 3.9 as the base image
FROM python:3.9-slim

# Copy requirements.txt from the current directory to the root of the docker image
COPY requirements.txt /

# Install the Python packages specified in requirements.txt
RUN pip install -r requirements.txt

# Copy the current directory's content to the /app folder in the docker image
COPY . /app

# Expose port 5000 for the app to be accessible
EXPOSE 5000

# Set an environment variable 'PORT' to be used by the application
ENV PORT 5000

# Set the working directory inside the container to /app
WORKDIR /app

# Use gunicorn as the application server. 
# This command will be run when the container starts.
# vec_eval_text:app is the Python module and Flask app to run.
# It uses a single worker, a single thread and a timeout of 60 seconds.
CMD exec gunicorn --bind :$PORT vec_eval_text:app --workers 1 --threads 1 --timeout 60
