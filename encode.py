#!/usr/bin/env python3
# Usage:
#  PYTHONPATH=src ./encode.py <file|directory|glob> /path/to/output.npz
#  PYTHONPATH=src ./train --dataset /path/to/output.npz

import fire
import json
import os
import numpy as np
import tensorflow as tf
import random
import time
import tqdm
import glob

import encoder


def load_dataset(enc, path):
    paths = []
    if os.path.isfile(path):
        # Simple file
        paths.append(path)
    elif os.path.isdir(path):
        # Directory
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                paths.append(os.path.join(dirpath, fname))
    else:
        # Assume glob
        paths = glob.glob(path)

    token_chunks = []
    for path in tqdm.tqdm(paths):
        with open(path, 'r') as fp:
            raw_text = fp.read()
        tokens = np.stack(enc.encode(raw_text))
        token_chunks.append(tokens)
    return token_chunks


def encode_main(in_text, out_npz, model_name='117M'):
    enc = encoder.get_encoder(model_name)
    print('Reading files')
    chunks = load_dataset(enc, in_text)
    print('Writing', out_npz)
    np.savez_compressed(out_npz, *chunks)


if __name__ == '__main__':
    fire.Fire(encode_main)
