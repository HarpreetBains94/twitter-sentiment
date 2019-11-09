FROM tensorflow/tensorflow:1.14.0-py3
COPY app /app
WORKDIR /app
RUN pip3 install -r requirements.txt
EXPOSE 5000
CMD ["python3", "api.py"]