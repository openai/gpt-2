FROM tensorflow/tensorflow:1.12.0-gpu-py3

# nvidia-docker 1.0
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=8.0" \
    LANG=C.UTF-8

RUN mkdir /gpt-2
WORKDIR /gpt-2
COPY requirements.txt download_model.sh /gpt-2/
RUN apt-get update && \
    apt-get install -y curl && \
    sh download_model.sh 117M
RUN pip3 install -r requirements.txt

ADD . /gpt-2
