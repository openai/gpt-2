#!/bin/bash

sudo docker build -f Dockerfile.train -t gpt2-train .
sudo docker run --gpus all -it -v $(pwd)/checkpoints_from_scratch:/gpt-2/checkpoints_from_scratch gpt2-train
