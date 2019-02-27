# gpt-2

Code and samples from the paper ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf).

For now, we have only released a smaller (117M parameter) version of GPT-2.

See more details in our [blog post](https://blog.openai.com/better-language-models/).

## Installation

Git clone this repository, and `cd` into directory for remaining commands
```
git clone https://github.com/openai/gpt-2.git && cd gpt-2
```

Then, follow instructions for either native or Docker installation.

### Native Installation

Download the model data
```
sh download_model.sh 117M
```

The remaining steps can optionally be done in a virtual environment using tools such as `virtualenv` or `conda`.

Install tensorflow 1.12 (with GPU support, if you have a GPU and want everything to run faster)
```
pip3 install tensorflow==1.12.0
```
or
```
pip3 install tensorflow-gpu==1.12.0
```

Install other python packages:
```
pip3 install -r requirements.txt
```

### Docker Installation

Build the Dockerfile and tag the created image as `gpt-2`:
```
docker build --tag gpt-2 -f Dockerfile.gpu . # or Dockerfile.cpu
```

Start an interactive bash session from the `gpt-2` docker image.

You can opt to use the `--runtime=nvidia` flag if you have access to a NVIDIA GPU
and a valid install of [nvidia-docker 2.0](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)).
```
docker run --runtime=nvidia -it gpt-2 bash
```

## Usage

| WARNING: Samples are unfiltered and may contain offensive content. |
| --- |

Some of the examples below may include Unicode text characters. Set the environment variable:
```
export PYTHONIOENCODING=UTF-8
```
to override the standard stream settings in UTF-8 mode.

### Unconditional sample generation

To generate unconditional samples from the small model:
```
python3 src/generate_unconditional_samples.py | tee /tmp/samples
```
There are various flags for controlling the samples:
```
python3 src/generate_unconditional_samples.py --top_k 40 --temperature 0.7 | tee /tmp/samples
```

### Conditional sample generation

To give the model custom prompts, you can use:
```
python3 src/interactive_conditional_samples.py --top_k 40
```

## GPT-2 samples

| WARNING: Samples are unfiltered and may contain offensive content. |
| --- |

While we have not yet released GPT-2 itself, you can see some samples from it in the `gpt-2-samples` folder.
We show unconditional samples with default settings (temperature 1 and no truncation), with temperature 0.7, and with truncation with top_k 40.
We show conditional samples, with contexts drawn from `WebText`'s test set, with default settings (temperature 1 and no truncation), with temperature 0.7, and with truncation with top_k 40.

## Future work

We may release code for evaluating the models on various benchmarks.

We are still considering release of the larger models.
