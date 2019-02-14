# gpt-2

Code and samples from the paper "Language Models are Unsupervised Multitask Learners"

## Installation

Download the (small, 117M parameter) model data:
```
gsutil rsync -r gs://gpt-2/models/ models/
```

Install python packages:
```
pip install -r requirements.txt
```

## Sample generation

| WARNING: Samples are unfiltered and may contain offensive content. |
| --- |

To generate unconditional samples from the small model:
```
python3 src/main.py | tee samples
```
There are various flags for controlling the samples:
```
python3 src/main.py --top_k 40 --temperature 0.7 | tee samples
```

While we have not yet released GPT-2 itself, you can see some unconditional samples (with default settings of temperature 1 and no truncation) in `gpt2-samples.txt`.

## Future work

We may release code for evaluating the models on various benchmarks.

We are still considering release of the larger models.
