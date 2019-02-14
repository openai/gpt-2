FROM tensorflow/tensorflow:1.12.0-py3

ENV LANG=C.UTF-8
RUN mkdir /gpt-2 
WORKDIR /gpt-2
COPY requirements.txt download_model.sh /gpt-2/
RUN apt-get update && \
    apt-get install -y curl && \
    sh download_model.sh 117M
RUN pip3 install -r requirements.txt

ADD . /gpt-2
