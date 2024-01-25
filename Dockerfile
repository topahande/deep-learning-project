FROM python:3.9-slim

RUN pip install --upgrade pip

RUN pip install flask

RUN pip install gunicorn

RUN pip install requests

RUN pip install keras-image-helper

RUN pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime

WORKDIR /app

COPY ["predict.py", "fruit-model.tflite", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]

