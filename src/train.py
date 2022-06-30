import argparse
import os
import sys

import lib_logging
from build_tokenizer import TokenizerConfig

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
        '--tokenizer-index',
        help="Path to tokenizer input file which specifies which tokenizer to use",
        type=str,
        default=os.path.realpath(os.path.join(PROG_DIR, "../training-data/tokenizer-index.json")),
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
    
    tokenizer_config = TokenizerConfig.load(args.tokenizer_index)

    # Build dataset
    (data, config) = load_dataset(
        tokenizer_config=tokenizer_config,
        dataset=args.dataset,
    )

    # Train
    train(
        data=data,
        config=config,
        tokenizer_config=tokenizer_config,
        gpu=args.gpu,
    )

def load_dataset(
    tokenizer_config: TokenizerConfig,
    dataset: str,
):
    # Check if encoded cache exists
    encoded_dataset_path = os.path.splitext(dataset)[0] + '.tar.gz'
    dataset_kwargs = {}

    if os.path.exists(encoded_dataset_path):
        dataset_kwargs['from_cache'] = encoded_dataset_path
    
    # Load dataset
    data = TokenDataset(
        file_path=dataset,
        vocab_file=tokenizer_config.vocab_file,
        merges_file=tokenizer_config.merges_file,
        #**dataset_kwargs,
    )
    config = build_gpt2_config(vocab_size=tokenizer_config.vocab_size)

    return (data, config)

def train(
    data,
    config,
    tokenizer_config: TokenizerConfig,
    gpu: bool,
):
    model = aitextgen(
        tokenizer_file=tokenizer_config.tokenizer_model_overview_file,
        config=config,
        to_gpu=gpu,
    )

    try:
        model.train(data, num_workers=1)
    except Exception as e:
        logger.error("Failed to train model", e)
        sys.exit(1)

if __name__ == '__main__':
    main()