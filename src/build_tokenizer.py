import argparse
import os
import json
from typing import Optional, List

from tokenizers import ByteLevelBPETokenizer

import lib_logging

logger = lib_logging.make_logger('build-tokenizer')

PROG_DIR = os.path.dirname(os.path.realpath(__file__))

class TokenizerConfig:
    """ Holds parameters and information for a tokenizer.
    Fields:
    - name: Identifying name of tokenizer
    - parent_dir: Directory in which files are saved
    - tokenizer_model_overview_file: Path to JSON file which contains overview of encoding
    - vocab_file: Path to text file with tokenizer vocabulary words
    - merges_file: Path to text file containing tokenizer merges
    - vocab_size: Number of words in the tokenizer's vocabulary
    """
    name: str
    parent_dir: str
    tokenizer_model_overview_file: str
    vocab_file: str
    merges_file: str
    vocab_size: int

    def __init__(
        self,
        parent_dir: str,
        name: str,
        vocab_size: int,
        tokenizer_model_overview_file: Optional[str]=None,
        vocab_file: Optional[str]=None,
        merges_file: Optional[str]=None,
    ):
        """ Initializes a TokenizerConfig.
        Arguments:
        - name: See TokenizerConfig.name
        - parent_dir: See TokenizerConfig.parent_dir
        - vocab_size: See TokenizerConfig.vocab_size
        - tokenizer_model_overview_file, vocab_file, merges_file: Allows parameters to be manually specified, if not specified default values are generated
        """
        self.name = name
        self.parent_dir = parent_dir
        self.vocab_size = vocab_size

        self.tokenizer_model_overview_file = tokenizer_model_overview_file or os.path.join(parent_dir, f"{name}-model-overview.json")
        self.vocab_file = vocab_file or os.path.join(parent_dir, f"{name}-vocab.json")
        self.merges_file = merges_file or os.path.join(parent_dir, f"{name}-merges.txt")

    def save(self, out_path: str):
        """ Saves the TokenizerConfig to a file.
        Arguments:
        - out_path: Path to JSON file where contents will be stored
        """
        with open(out_path, 'w') as out_f:
            json.dump({
                'name': self.name,
                'parent_dir': self.parent_dir,
                'vocab_size': self.vocab_size,
                'tokenizer_model_overview_file': self.tokenizer_model_overview_file,
                'vocab_file': self.vocab_file,
                'merges_file': self.merges_file,
            }, out_f)

    @staticmethod
    def load(load_path: str) -> 'TokenizerConfig':
        """ Loads a TokenizerConfig from a file.
        Arguments:
        - load_path: Path from which to load config

        Returns: TokenizerConfig stored in file

        Raises:
        - KeyError: If a required TokenizerConfig is not present in the file
        """
        with open(load_path, 'r') as load_f:
            data = json.load(load_f)

            return TokenizerConfig(
                name=data['name'],
                parent_dir=data['parent_dir'],
                vocab_size=data['vocab_size'],
                tokenizer_model_overview_file=data['tokenizer_model_overview_file'],
                vocab_file=data['vocab_file'],
                merges_file=data['merges_file'],
            )

def parse_args():
    """ Parse command line arguments.
    Returns: Argument values
    """
    parser = argparse.ArgumentParser(description="Given a dataset creates an appropriate tokenizer")

    parser.add_argument(
        '--dataset-in',
        help="Path to training data input file",
        type=str,
        default=os.path.realpath(os.path.join(PROG_DIR, "../training-data/discord-messages.txt")),
    )

    parser.add_argument(
        '--tokenizer-index-out',
        help="Path of JSON file to save tokenizer configuration values",
        type=str,
        default=os.path.realpath(os.path.join(PROG_DIR, "../training-data/tokenizer-index.json"))
    )

    parser.add_argument(
        '--tokenizer-out-dir',
        help="Path to directory where tokenizer results files will be saved",
        type=str,
        default=os.path.realpath(os.path.join(PROG_DIR, "../training-data")),
    )
    parser.add_argument(
        '--tokenizer-name',
        help="Name of tokenizer, output files will be prefixed with this value",
        type=str,
        default="tokenizer",
    )

    parser.add_argument(
        '--vocab-size',
        help="Number of words allowed in the vocabulary",
        type=int,
        default=50_000,
    )

    parser.add_argument(
        '--sample-start-token',
        help="Token placed before a sample",
        type=str,
        default="<|endoftext|>",
    )
    parser.add_argument(
        '--sample-end-token',
        help="Token placed after a sample",
        type=str,
        default="<|endoftext|>",
    )

    # Parse results
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    tokenizer_config = TokenizerConfig(
        name=args.tokenizer_name,
        parent_dir=args.tokenizer_out_dir,
        vocab_size=args.vocab_size,
    )

    # Train tokenizer
    train_tokenizer(
        dataset_in=args.dataset_in,
        tokenizer_config=tokenizer_config,
        sample_start_token=args.sample_start_token,
        sample_end_token=args.sample_end_token,
    )

    tokenizer_config.save(args.tokenizer_index_out)
    logger.info(f"Saved tokenizer index file to '{args.tokenizer_index_out}'")

def train_tokenizer(
    dataset_in: str,
    tokenizer_config: TokenizerConfig,
    sample_start_token: str,
    sample_end_token: str,
):
    tokenizer = ByteLevelBPETokenizer(
        dropout=None,
        trim_offsets=True,
    )

    tokenizer.train(
        files=[dataset_in],
        vocab_size=tokenizer_config.vocab_size,
        min_frequency=2,
        special_tokens=[
            sample_start_token,
            sample_end_token,
        ]
    )
    
    tokenizer.save(tokenizer_config.tokenizer_model_overview_file)
    tokenizer.save_model(tokenizer_config.parent_dir, tokenizer_config.name)

    logger.info(f"Encoded tokenizer '{tokenizer_config.name}' on '{dataset_in}' and saved to '{tokenizer_config.parent_dir}'")


if __name__ == '__main__':
    main()