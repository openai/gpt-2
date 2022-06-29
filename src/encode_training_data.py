import argparse
import os

from tokenizers import ByteLevelBPETokenizer

import lib_logging

logger = lib_logging.make_logger('encode-training-data')

PROG_DIR = os.path.dirname(os.path.realpath(__file__))

class TokenizerConfigLocation:
    """ Indicates location of tokenizer configuration files.
    Fields:
    - name: Identifying name of tokenizer
    - parent_dir: Directory in which files are saved
    - tokenizer_file: Path to JSON file which contains overview of encoding
    - vocab_file: Path to text file with tokenizer vocabulary words
    - merges_file: Path to text file containing tokenizer merges
    """
    name: str
    parent_dir: str
    tokenizer_file: str
    vocab_file: str
    merges_file: str

    def __init__(self, parent_dir: str, name: str):
        """ Initializes a TokenizerConfigLocation.
        Arguments:
        - dir: Directory in which tokenizer files will be stored
        - name: Identifier of tokenizer, will be included in file names
        """
        self.name = name
        self.parent_dir = parent_dir
        self.tokenizer_file = os.path.join(parent_dir, f"{name}.json")
        self.vocab_file = os.path.join(parent_dir, f"{name}-vocab.json")
        self.merges_file = os.path.join(parent_dir, f"{name}-merges.txt")

def parse_args():
    parser = argparse.ArgumentParser(description="Encodes training data")
    parser.add_argument(
        '--training-data-in',
        help="Path to training data input file",
        type=str,
        default=os.path.realpath(os.path.join(PROG_DIR, "../training-data/discord-messages.txt")),
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

    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Encode
    loc_config = TokenizerConfigLocation(args.tokenizer_out_dir, args.tokenizer_name)

    encode_training_data(
        training_data_in=args.training_data_in,
        tokenizer_location_config=loc_config,
        vocab_size=args.vocab_size,
        sample_start_token=args.sample_start_token,
        sample_end_token=args.sample_end_token,
    )

def encode_training_data(
    training_data_in: str,
    tokenizer_location_config: TokenizerConfigLocation,
    vocab_size: int,
    sample_start_token: str,
    sample_end_token: str,
):
    tokenizer = ByteLevelBPETokenizer(
        dropout=None,
        trim_offsets=True,
    )

    tokenizer.train(
        files=[training_data_in],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[
            sample_start_token,
            sample_end_token,
        ]
    )
    
    tokenizer.save(tokenizer_location_config.tokenizer_file)
    tokenizer.save_model(tokenizer_location_config.parent_dir, tokenizer_location_config.name)

    logger.info(f"Encoded tokenizer '{tokenizer_location_config.name}' on '{training_data_in}' and saved to '{tokenizer_location_config.parent_dir}'")

if __name__ == '__main__':
    main()