#!/usr/bin/env python3
# Usage:
#  PYTHONPATH=src ./encode.py <file|directory|glob> /path/to/output.npz
#  PYTHONPATH=src ./train --dataset /path/to/output.npz

import argparse
import numpy as np

import encoder
from load_dataset import load_dataset

parser = argparse.ArgumentParser(
    description='Pre-encode text files into tokenized training set.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_name', metavar='MODEL', type=str, default='117M', help='Pretrained model name')
parser.add_argument('--combine', metavar='CHARS', type=int, default=50000, help='Concatenate files with <|endoftext|> separator into chunks of this minimum size')
parser.add_argument('in_text', metavar='PATH', type=str, help='Input file, directory, or glob pattern (utf-8 text).')
parser.add_argument('out_npz', metavar='OUT.npz', type=str, help='Output file path')

def main():
    args = parser.parse_args()
    enc = encoder.get_encoder(args.model_name)
    print('Reading files')
    chunks = load_dataset(enc, args.in_text, args.combine)
    print('Writing', args.out_npz)
    np.savez_compressed(args.out_npz, *chunks)


if __name__ == '__main__':
    main()
