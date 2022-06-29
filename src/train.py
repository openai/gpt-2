import argparse
import os

import lib_logging

logger = lib_logging.make_logger('train')

PROG_DIR = os.path.dirname(os.path.realpath(__file__))

from aitextgen import aitextgen
from aitextgen.TokenDataset import TokenDataset
from aitextgen.utils import build_gpt2_config

def parse_args():
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument(
        '--gpu',
        help="Train the model on a GPU",
        action='store_true',
    )

    parser.add_argument(
        '--tokenizer-in',
        help="Path to tokenizer configuration file",
        type=str,
        default=os.path.realpath(os.path.join(PROG_DIR, "../training-data/tokenizer.json")),
    )
    parser.add_argument(
        '--vocab-size',
        help="Number of words in the tokenizer vocabulary",
        type=int,
        default=50_000,
    )

    parser.add_argument(
        '--dataset',
        help="Path to training dataset file",
        type=str,
        default=os.path.realpath(os.path.join(PROG_DIR, "../training-data/discord-messages.txt")),
    )

    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Build dataset
    (data, config) = load_dataset(
        tokenizer_in=args.tokenizer_in,
        vocab_size=args.vocab_size,
        dataset=args.dataset,
    )
    # Train
    train(
        data=data,
        config=config,
        tokenizer_in=args.tokenizer_in,
        gpu=args.gpu,
    )

def load_dataset(
    tokenizer_in: str,
    vocab_size: int,
    dataset: str,
):
    data = TokenDataset(
        file_path=dataset,
    )
    config = build_gpt2_config(vocab_size=vocab_size)

    return (data, config)

def train(
    data,
    config,
    tokenizer_in: str,
    gpu: bool,
):
    model = aitextgen(
        tokenizer_file=tokenizer_in,
        config=config,
    )

    if gpu:
        model.to_gpu()
    
    model.train(data)

if __name__ == '__main__':
    main()