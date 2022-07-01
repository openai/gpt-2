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
        '--prompt-template-file',
        help="Location of text file which will be used as a template to insert the prompt into, the text '<PROMPT>' will be replaced with the prompt"
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

    logger.info(f"Loaded model, trained for {training_meta.training_iterations} iterations")

    # Check for prompt template file
    prompt_template = lambda prompt: prompt
    if args.prompt_template_file:
        with open(args.prompt_template_file, 'r') as prompt_template_f:
            txt = prompt_template_f.read()
            prompt_template = lambda prompt: txt.replace('<PROMPT>', prompt)

    # Evaluate model
    if args.prompt is not None:
        prompt = prompt_template(args.prompt)

        results = evaluate_prompt(
            model=model,
            prompt=args.prompt,
            max_results=args.max_results,
            remove_prompt=args.prompt_template_file is not None,
        )

        print_results(
            prompt=prompt,
            results=results,
        )
    elif args.interactive_prompt:
        while True:
            prompt = prompt_template(input("Prompt: ").strip())

            results = evaluate_prompt(
                model=model,
                prompt=prompt,
                max_results=args.max_results,
                remove_prompt=args.prompt_template_file is not None,
            )

            print_results(
                prompt=prompt,
                results=results,
            )

def print_results(
    prompt: str,
    results: List[str],
):
    """ Given a list of model results print them to the console.
    Arguments:
    - prompt: Text which was inputted into the model to obtain results
    - results: Model evaluation results to print
    """
    bar = "=" * 20

    # Show prompt
    logger.info(f"{bar} Prompt {bar}")
    logger.info(prompt)

    # Print results
    i = 1
    for result in results:
        logger.info(f"{bar} Result {i} {bar}")
        logger.info(result)
        i += 1

    logger.info("=" * ((len(bar) * 2) + len(" Result n ")))

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
    remove_prompt: bool,
) -> List[str]:
    """ Given an input prompt evaluate the model and return results.
    Arguments:
    - model: The model to evalute prompt against
    - prompt: Input text for model evaluation
    - max_results: The maximum number of results which can be returned
    - remove_prompt: If True then any text which matches the prompt will be removed from results

    Returns: List of results no greater than max_results long.
    """
    results = model.generate(
        n=max_results,
        prompt=prompt,
        return_as_list=True,
        nonempty_output=True,
    )

    # Remove the prompt from results
    if remove_prompt:
        results = [ result.replace(prompt, "") for result in results ]

    # Remove empty results
    results = filter(lambda result: len(result) > 0, results)


    return results

if __name__ == '__main__':
    main()