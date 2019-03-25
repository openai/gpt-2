# gpt-2

Code and samples from the paper ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf).

For now, we have only released a smaller (117M parameter) version of GPT-2.

See more details in our [blog post](https://blog.openai.com/better-language-models/).

## Usage

This repository is meant to be a starting point for researchers and engineers to experiment with GPT-2-117M.  While GPT-2-117M is less proficient than GPT-2-1.5B, it is useful for a wide range of research and applications which could also apply to larger models.

### Some caveats

- GPT-2-117M robustness and worst case behaviors are not well-understood.  As with any machine-learned model, carefully evaluate GPT-2-117M for your use case, especially if used without fine-tuning or in safety-critical applications where reliability is important.
- The dataset our GPT-2-117M was trained on contains many texts with [biases](https://twitter.com/TomerUllman/status/1101485289720242177) and factual inaccuracies, and thus GPT-2-117M is likely to be biased and inaccurate as well.
- To avoid having samples mistaken as human-written, we recommend clearly labeling samples as synthetic before wide dissemination.  Our models are often incoherent or inaccurate in subtle ways, which takes more than a quick read for a human to notice.

### Work with us

Please [let us know](mailto:languagequestions@openai.com) if you’re doing interesting research with or working on applications of GPT-2-117M!  We’re especially interested in hearing from and potentially working with those who are studying
- Potential malicious use cases and defenses against them (e.g. the detectability of synthetic text)
- The extent of problematic content (e.g. bias) being baked into the models and effective mitigations

## Development

See [DEVELOPERS.md](./DEVELOPERS.md)

## Contributors

See [CONTRIBUTORS.md](./CONTRIBUTORS.md)

### Fine tuning on custom datasets

To retrain GPT-2 117M model on a custom text dataset:

```
PYTHONPATH=src ./train.py --dataset <file|directory|glob>
```

If you want to precompute the dataset's encoding for multiple runs, you can instead use:

```
PYTHONPATH=src ./encode.py <file|directory|glob> /path/to/encoded.npz
PYTHONPATH=src ./train.py --dataset /path/to/encoded.npz
```

To do distributed on multiple GPUs or machines using Horovod: 

```
mpirun -np 4 \
    -H localhost:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -x PYTHONPATH=src \
    -mca pml ob1 -mca btl ^openib \
    /home/jovyan/gpt-2/train-horovod.py --dataset encoded.npz
```

## GPT-2 samples

| WARNING: Samples are unfiltered and may contain offensive content. |
| --- |

While we have not yet released GPT-2 itself, you can see some samples from it in the `gpt-2-samples` folder.
We show unconditional samples with default settings (temperature 1 and no truncation), with temperature 0.7, and with truncation with top_k 40.
We show conditional samples, with contexts drawn from `WebText`'s test set, with default settings (temperature 1 and no truncation), with temperature 0.7, and with truncation with top_k 40.

## Citation

Please use the following bibtex entry:
```
@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  year={2019}
}
```

## Future work

We may release code for evaluating the models on various benchmarks.

We are still considering release of the larger models.

## License

[MIT](./LICENSE)
