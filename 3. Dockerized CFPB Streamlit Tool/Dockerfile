FROM python:3.9-slim
# Copy requirements.txt to the docker image and install packages
COPY requirements.txt /
RUN pip install -r requirements.txt
# Set the WORKDIR to be the folder
COPY . /app
EXPOSE 8501
ENV PORT 8501
WORKDIR /app
# Use gunicorn as the entrypoint
CMD ["/bin/bash", "wrapper_script.sh"]