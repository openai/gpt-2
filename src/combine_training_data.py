import json
import os
import glob
import argparse

import lib_logging
from lib_path import LocalPath

logger = lib_logging.make_logger('combine-training-data')

def parse_args():
    parser = argparse.ArgumentParser(description="Combine Discord messages into a training data set file.")
    parser.add_argument(
        '--discord-messages-dir',
        help="Directory in which Discord channel message JSON dumps are stored",
        type=LocalPath,
        default=LocalPath("discord-messages"),
    )
    parser.add_argument(
        '--training-data-out',
        help="File in which training data will be outputted",
        type=LocalPath,
        default=LocalPath("training-data/discord-messages.txt"),
    )

    parser.add_argument(
        '--prepend-authors',
        help="Place the name of the author before each message",
        action='store_true',
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

    # Combine training data
    combine_training_data(
        discord_messages_dir=args.discord_messages_dir,
        training_data_out=args.training_data_out,
        prepend_authors=args.prepend_authors,
        sample_start_token=args.sample_start_token,
        sample_end_token=args.sample_end_token,
    )

def combine_training_data(
    discord_messages_dir: LocalPath,
    training_data_out: LocalPath,
    prepend_authors: bool,
    sample_start_token: str,
    sample_end_token: str,
):
    """ Given a directory of Discord channel message JSON dumps combine them into a single text file full of samples with which the model can be trained.
    Arguments:
    - discord_messages_dir: Directory in which Discord channel message JSON dumps are contained
    - training_data_out: Path where combined training data text file will be saved
    - prepend_authors: If True the author of each Discord message will be prepended before each sample
    - sample_start_token: Token used to delineate the start of a sample
    - sample_end_token: Token used to delineate the end of a sample
    """
    # Make directory in which to output training data
    training_data_out_dir = os.path.dirname(training_data_out.get_absolute_path())
    if not os.path.isdir(training_data_out_dir):
        os.makedirs(training_data_out_dir)

    # Combine
    def extract_discord_msg(msg):
        """ Based on the arguments to combine_training_data make a Discord message dump into a string.
        Arguments:
        - msg: Discord message dump object
        """
        out = []

        # Start token
        out.append(sample_start_token)
        
        # Author
        if prepend_authors:
            out.extend([
                msg['author']['name'],
                "#",
                msg['author']['discriminator'],
                ": ",
            ])

        # Content
        out.append(msg['content'])

        # End token
        out.append(sample_end_token)

        # Done
        return "".join(out)

    num_msgs = 0
    with open(training_data_out.get_absolute_path(), "w", encoding="utf8") as training_data_f:
        # For each discord channel message JSON dump file
        for discord_msg_file_path in glob.glob(f"{discord_messages_dir}/*"):
            with open(discord_msg_file_path, "r", encoding="utf8") as discord_msg_f:
                discord_msg_json = json.load(discord_msg_f)

                # Process messages
                msgs = [
                    extract_discord_msg(msg)
                    for msg in filter(lambda msg: len(msg['content']) > 0, discord_msg_json['messages'])
                ]

                # Write to file
                training_data_f.writelines("\n".join(msgs))
            
                # Record statistics
                num_msgs += len(msgs)
                logger.info(f"Wrote {len(msgs)} from {discord_msg_file_path} to training file")

    logger.info(f"Wrote {num_msgs} to {training_data_out}")

if __name__ == '__main__':
    main()
