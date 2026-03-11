FROM python:3.8-slim-buster

RUN apt update -y && apt install awscli -y
WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Expose port
EXPOSE 8080

CMD ["python3", "app.py"]
