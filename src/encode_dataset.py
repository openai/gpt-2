import argparse
import os
import json
from typing import Optional, List

from aitextgen.TokenDataset import TokenDataset

import lib_logging
from lib_path import LocalPath
from build_tokenizer import TokenizerConfig

logger = lib_logging.make_logger('encode-dataset')

def parse_args():
    """ Parse command line arguments.
    Returns: Argument values
    """
    parser = argparse.ArgumentParser(description="Encodes dataset using tokenizer")

    parser.add_argument(
        '--dataset-in',
        help="Path to training data input file",
        type=LocalPath,
        default=LocalPath("training-data/discord-messages.txt"),
    )
    parser.add_argument(
        '--dataset-out',
        help="Path to which encoded training data will be saved",
        type=LocalPath,
        default=LocalPath("training-data/discord-messages.tar.gz"),
    )
    parser.add_argument(
        '--tokenizer-index',
        help="Path to tokenizer input file which specifies which tokenizer to use",
        type=LocalPath,
        default=LocalPath("training-data/tokenizer-index.json"),
    )

    # Parse results
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    tokenizer_config = TokenizerConfig.load(args.tokenizer_index)

    # Encode dataset
    encode_dataset(
        dataset_in=args.dataset_in,
        dataset_out=args.dataset_out,
        tokenizer_config=tokenizer_config,
    )

def encode_dataset(
    dataset_in: LocalPath,
    dataset_out: LocalPath,
    tokenizer_config: TokenizerConfig,
):
    """ Given a dataset encodes the contents using the tokenizer. Saves the results.
    Arguments:
    - dataset_in: Path to the un-encoded dataset file
    - dataset_out: Path where the encoded dataset will be saved
    - tokenizer_config: Information about the tokenizer
    """
    TokenDataset(
        file_path=dataset_in.get_absolute_path(),
        vocab_file=tokenizer_config.vocab_file.get_absolute_path(),
        merges_file=tokenizer_config.merges_file.get_absolute_path(),
        save_cache=True,
        cache_destination=dataset_out.get_absolute_path(),
    )

if __name__ == '__main__':
    main()
