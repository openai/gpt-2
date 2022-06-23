import json
import os
import glob

PROG_DIR = os.path.dirname(os.path.realpath(__file__))
DISCORD_MESSAGES_DIR = os.path.join(PROG_DIR, "../discord-messages")
TRAINING_DATA_DIR = os.path.join(PROG_DIR, "../training-data")
TRAINING_DATA_FILE = os.path.join(TRAINING_DATA_DIR, "discord-messages.txt")

TRAINING_DELIMITER = "\n<|endoftext|>\n"

DISCORD_MSG_FILES = glob.glob(f"{DISCORD_MESSAGES_DIR}/*.json")

if not os.path.isdir(TRAINING_DATA_DIR):
    os.makedirs(TRAINING_DATA_DIR)

num_msgs = 0
with open(TRAINING_DATA_FILE, "w") as training_data_f:
    for discord_msg_file_path in DISCORD_MSG_FILES:
        with open(discord_msg_file_path, "r") as discord_msg_f:
            discord_msg_json = json.load(discord_msg_f)

            msgs = list(filter(lambda t: len(t) > 0, [ msg['content'] for msg in discord_msg_json['messages'] ]))
            training_data_f.writelines(TRAINING_DELIMITER.join(msgs))
            
            num_msgs += len(msgs)
            print(f"Wrote {len(msgs)} from {discord_msg_file_path} to training file")

print(f"Wrote {num_msgs} to {TRAINING_DATA_FILE}")
