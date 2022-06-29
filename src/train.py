import argparse
import os

import lib_logging
from encode_training_data import TokenizerConfigLocation

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
        default=False,
    )

    parser.add_argument(
        '--tokenizer-dir',
        help="Path to tokenizer configuration directory",
        type=str,
        default=os.path.realpath(os.path.join(PROG_DIR, "../training-data")),
    )
    parser.add_argument(
        '--tokenizer-name',
        help="Name of tokenizer, configuration files for the tokenizer are prefixed with this name",
        type=str,
        default="tokenizer",
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
    tokenizer_config_location = TokenizerConfigLocation(args.tokenizer_dir, args.tokenizer_name)

    (data, config) = load_dataset(
        tokenizer_config_location=tokenizer_config_location,
        vocab_size=args.vocab_size,
        dataset=args.dataset,
    )
    # Train
    train(
        data=data,
        config=config,
        tokenizer_config_location=tokenizer_config_location,
        gpu=args.gpu,
    )

def load_dataset(
    tokenizer_config_location: TokenizerConfigLocation,
    vocab_size: int,
    dataset: str,
):
    data = TokenDataset(
        file_path=dataset,
        vocab_file=tokenizer_config_location.vocab_file,
        merges_file=tokenizer_config_location.merges_file,
    )
    config = build_gpt2_config(vocab_size=vocab_size)

    return (data, config)

def train(
    data,
    config,
        tokenizer_config_location: TokenizerConfigLocation,
    gpu: bool,
):
    model = aitextgen(
        tokenizer_file=tokenizer_config_location.tokenizer_file,
        config=config,
        to_gpu=gpu,
    )

    model.train(data)

if __name__ == '__main__':
    main()