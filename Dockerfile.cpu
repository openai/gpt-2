FROM tensorflow/tensorflow:1.12.0-py3

ENV LANG=C.UTF-8
RUN mkdir /gpt-2
WORKDIR /gpt-2
ADD . /gpt-2
RUN pip3 install -r requirements.txt
RUN python3 download_model.py 117M
RUN python3 download_model.py 345M
