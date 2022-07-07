import argparse
import os
import json
from typing import Optional, List

from tokenizers import ByteLevelBPETokenizer

import lib_logging
from lib_path import LocalPath

logger = lib_logging.make_logger('build-tokenizer')

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
    parent_dir: LocalPath
    tokenizer_model_overview_file: LocalPath
    vocab_file: LocalPath
    merges_file: LocalPath
    vocab_size: int

    def __init__(
        self,
        parent_dir: LocalPath,
        name: str,
        vocab_size: int,
        tokenizer_model_overview_file: Optional[LocalPath]=None,
        vocab_file: Optional[LocalPath]=None,
        merges_file: Optional[LocalPath]=None,
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

        self.tokenizer_model_overview_file = tokenizer_model_overview_file or self.parent_dir.join([f"{name}-model-overview.json"])
        self.vocab_file = vocab_file or self.parent_dir.join([f"{name}-vocab.json"])
        self.merges_file = merges_file or self.parent_dir.join([f"{name}-merges.txt"])

    def save(self, out_path: str):
        """ Saves the TokenizerConfig to a file.
        Arguments:
        - out_path: Path to JSON file where contents will be stored
        """
        with open(out_path, 'w') as out_f:
            json.dump(
                {
                    'name': self.name,
                    'parent_dir': self.parent_dir.get_project_relative_path(),
                    'vocab_size': self.vocab_size,
                    'tokenizer_model_overview_file': self.tokenizer_model_overview_file.get_project_relative_path(),
                    'vocab_file': self.vocab_file.get_project_relative_path(),
                    'merges_file': self.merges_file.get_project_relative_path(),
                },
                out_f,
                indent=4,
            )

    @staticmethod
    def load(load_path: LocalPath) -> 'TokenizerConfig':
        """ Loads a TokenizerConfig from a file.
        Arguments:
        - load_path: Path from which to load config

        Returns: TokenizerConfig stored in file

        Raises:
        - KeyError: If a required TokenizerConfig is not present in the file
        """
        with open(load_path.get_absolute_path(), 'r') as load_f:
            data = json.load(load_f)

            return TokenizerConfig(
                name=data['name'],
                parent_dir=LocalPath(data['parent_dir']),
                vocab_size=data['vocab_size'],
                tokenizer_model_overview_file=LocalPath(data['tokenizer_model_overview_file']),
                vocab_file=LocalPath(data['vocab_file']),
                merges_file=LocalPath(data['merges_file']),
            )

def parse_args():
    """ Parse command line arguments.
    Returns: Argument values
    """
    parser = argparse.ArgumentParser(description="Given a dataset creates an appropriate tokenizer")

    parser.add_argument(
        '--dataset-in',
        help="Path to training data input file",
        type=LocalPath,
        default=LocalPath("training-data/discord-messages.txt"),
    )

    parser.add_argument(
        '--tokenizer-index-out',
        help="Path of JSON file to save tokenizer configuration values",
        type=LocalPath,
        default=LocalPath("training-data/tokenizer-index.json"),
    )

    parser.add_argument(
        '--tokenizer-out-dir',
        help="Path to directory where tokenizer results files will be saved",
        type=LocalPath,
        default=LocalPath("training-data"),
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
    logger.info(f"Saved tokenizer index file to '{args.tokenizer_index_out.get_project_relative_path()}'")

def train_tokenizer(
    dataset_in: LocalPath,
    tokenizer_config: TokenizerConfig,
    sample_start_token: str,
    sample_end_token: str,
):
    """ Given a dataset generate a vocabulary and BPE encoding.
    Arguments:
    - dataset_in: Path to dataset file
    - tokenizer_config: Configuration for tokenizer properties and output
    - sample_start_token: Token used to delineate the start of a sample
    - sample_end_token: Token used to delineate the end of a sample
    """
    tokenizer = ByteLevelBPETokenizer(
        dropout=None,
        trim_offsets=True,
    )

    tokenizer.train(
        files=[dataset_in.get_absolute_path()],
        vocab_size=tokenizer_config.vocab_size,
        min_frequency=2,
        special_tokens=[
            sample_start_token,
            sample_end_token,
        ]
    )
    
    tokenizer.save(tokenizer_config.tokenizer_model_overview_file.get_absolute_path())
    tokenizer.save_model(tokenizer_config.parent_dir.get_absolute_path(), tokenizer_config.name)

    logger.info(f"Encoded tokenizer '{tokenizer_config.name}' on '{dataset_in.get_project_relative_path()}' and saved to '{tokenizer_config.parent_dir.get_project_relative_path()}'")


if __name__ == '__main__':
    main()
