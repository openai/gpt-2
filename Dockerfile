FROM tensorflow/tensorflow:1.15.2-py3-jupyter

# setup environment language
ENV LANG=C.UTF-8

# copy requirements.txt into image
COPY requirements.txt requirements.txt

# update and upgrade packages and pip and install python libraries
RUN apt-get update && apt-get upgrade -y \
&& apt-get install -y apt-utils \
&& pip3 install --upgrade pip \
&& pip3 install -r requirements.txt \
&& rm requirements.txt
