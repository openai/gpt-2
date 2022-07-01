import argparse
import os
import sys


from aitextgen import aitextgen
from aitextgen.TokenDataset import TokenDataset
from aitextgen.utils import build_gpt2_config
from transformers import GPT2Config 

import lib_logging
from build_tokenizer import TokenizerConfig

logger = lib_logging.make_logger('train')

PROG_DIR = os.path.dirname(os.path.realpath(__file__))

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

    parser.add_argument(
        '--models-dir',
        help="Parent directory where models will be saved",
        type=str,
        default="models",
    )
    parser.add_argument(
        '--model-name',
        help="Name of model, used to determine where it will be saved",
        type=str,
        default="model",
    )

    parser.add_argument(
        '--sample-every',
        help="The number of training steps which should occur between model output samples",
        type=int,
        default=1000,
    )

    parser.add_argument(
        '--target-epochs',
        help="Number of training epochs to run, -1 to continuously train with no limit",
        type=int,
        default=-1
    )

    parser.add_argument(
        '--train-epoch-steps',
        help="Number of steps of training to perform for each overall cycle of training",
        type=int,
        default=5000,
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
        output_parent_dir=args.models_dir,
        model_name=args.model_name,
        config=config,
        tokenizer_config=tokenizer_config,
        gpu=args.gpu,
        sample_every=args.sample_every,
        target_epochs=args.target_epochs,
        train_epoch_steps=args.train_epoch_steps,
    )

def load_dataset(
    tokenizer_config: TokenizerConfig,
    dataset: str,
) -> (TokenDataset, GPT2Config):
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
    data: TokenDataset,
    output_parent_dir: str,
    model_name: str,
    config: GPT2Config,
    tokenizer_config: TokenizerConfig,
    gpu: bool,
    sample_every: int,
    target_epochs: int,
    train_epoch_steps: int,
):
    model = aitextgen(
        tokenizer_file=tokenizer_config.tokenizer_model_overview_file,
        config=config,
        to_gpu=gpu,
    )

    output_dir = os.path.join(output_parent_dir, model_name)
    logger.info(f"Saving model into '{output_dir}' directory")

    
    num_epochs = 0
    total_steps = 0

    try:
        while num_epochs < target_epochs or target_epochs < 0:
            model.train(
                output_dir=output_dir,
                train_data=data,
                num_workers=1, # Required or else training on GPUs won't work
                generate_every=sample_every,
                num_steps=train_epoch_steps,
            )

            num_epochs += 1
            total_steps += train_epoch_steps

            logger.info(f"Completed training epoch {num_epochs}, total steps {total_steps}")
    except Exception as e:
        logger.error("Failed to train model", e)
        sys.exit(1)

    logger.info(f"Completed {num_epochs} of training, resulting in {total_steps} training iterations")
    logger.info(f"Model saved into '{output_dir}' directory")

if __name__ == '__main__':
    main()