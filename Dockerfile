FROM python:3.6-slim

COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install -y git python-dev python-pip && pip install -r requirements.txt && git clone https://www.github.com/keras-team/keras-contrib.git && cd keras-contrib && python setup.py install && python -m spacy download en_core_web_md

COPY ./ /app
WORKDIR /app