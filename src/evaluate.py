import argparse
import os
from typing import List

from aitextgen import aitextgen

import lib_logging
from train import TrainingMetadata
from build_tokenizer import TokenizerConfig

logger = lib_logging.make_logger('evaluate')

def parse_args():
    """ Parse command line arguments.
    Returns: Command line argument values.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument(
        '--models-dir',
        help="Parent directory where models are saved",
        type=str,
        default="models",
    )
    parser.add_argument(
        '--model-name',
        help="Name of model to evaluate",
        type=str,
        default="model",
    )

    prompt_parser = parser.add_mutually_exclusive_group(required=True)
    prompt_parser.add_argument(
        '--prompt',
        help="Text which will be provided to the model (cannot be provided with --interactive-prompt)",
        type=str,
    )
    prompt_parser.add_argument(
        '--interactive-prompt',
        help="Indicates the script should ask the user for prompts via stdin",
        action='store_true',
    )

    parser.add_argument(
        '--max-results',
        help="The maximum number of results which the model should return",
        type=int,
        default=5,
    )

    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Load the model
    model_dir = os.path.join(args.models_dir, args.model_name)
    model, training_meta = load_model(model_dir)

    # Evaluate model
    if args.prompt is not None:
        res = evaluate_prompt(
            model=model,
            prompt=args.prompt,
            max_results=args.max_results
        )
        print_results(res)
    elif args.interactive_prompt:
        while True:
            prompt = input("Prompt: ").strip()

            res = evaluate_prompt(
                model=model,
                prompt=prompt,
                max_results=args.max_results
            )
            print_results(res)

def print_results(
    results: List[str],
):
    """ Given a list of model results print them to the console.
    Arguments:
    - results: Model evaluation results to print
    """
    bar = "=" * 20
    i = 1
    for result in results:
        logger.info(f"{bar} Result {i} {bar}")
        logger.info(result)
        i += 1

def load_model(
    model_dir: str,
) -> (aitextgen, TrainingMetadata):
    """ Loads a trained model from the file system.
    Arguments:
    - model_dir: The directory in which the model data resides

    Returns: Loaded model.
    """
    # Load training metadata
    training_meta = TrainingMetadata.load(os.path.join(model_dir, "training-metadata.json"))
    tokenizer_config = TokenizerConfig.load(training_meta.tokenizer_index)

    # Load model
    model = aitextgen(
        model_folder=model_dir,
        tokenizer_file=tokenizer_config.tokenizer_model_overview_file,
    )

    return (model, training_meta)

def evaluate_prompt(
    model: aitextgen,
    prompt: str,
    max_results: int,
) -> List[str]:
    """ Given an input prompt evaluate the model and return results.
    Arguments:
    - model: The model to evalute prompt against
    - prompt: Input text for model evaluation
    - max_results: The maximum number of results which can be returned

    Returns: List of results no greater than max_results long.
    """
    return model.generate(
        n=max_results,
        prompt=prompt,
        return_as_list=True,
        nonempty_output=True,
    )

if __name__ == '__main__':
    main()