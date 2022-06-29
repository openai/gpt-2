import argparse
import os

from tokenizers import ByteLevelBPETokenizer

import lib_logging

logger = lib_logging.make_logger('encode-training-data')

PROG_DIR = os.path.dirname(os.path.realpath(__file__))

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
        help="Path to file where tokenizer results will be saved",
        type=str,
        default=os.path.realpath(os.path.join(PROG_DIR, "../training-data/tokenizer.json")),
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
    encode_training_data(
        training_data_in=args.training_data_in,
        tokenizer_out=args.tokenizer_out,
        vocab_size=args.vocab_size,
        sample_start_token=args.sample_start_token,
        sample_end_token=args.sample_end_token,
    )

def encode_training_data(
    training_data_in: str,
    tokenizer_out: str,
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
    
    tokenizer.save(tokenizer_out)

    logger.info(f"Encoded '{training_data_in}' and saved to '{tokenizer_out}'")

if __name__ == '__main__':
    main()