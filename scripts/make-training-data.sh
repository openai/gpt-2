#!/usr/bin/env bash
declare -r PROG_DIR=$(dirname $(realpath "$0"))

declare -r DOCKER_COMPOSE_SVC="model"

declare -r COMBINED_TRAINING_DATA_RELATIVE_TO_ROOT_PATH="training-data/discord-messages.txt"
declare -r COMBINED_TRAINING_DATA="$PROG_DIR/../$COMBINED_TRAINING_DATA_RELATIVE_TO_ROOT_PATH"

declare -r ENCODING_OUT_RELATIVE_TO_ROOT_PATH="training-data/discord-messages.npz"
declare -r ENCODING_OUT="$PROG_DIR/../$ENCODING_OUT_RELATIVE_TO_ROOT_PATH"


if ! [[ -f "$COMBINED_TRAINING_DATA" ]]; then
    echo "Combining all discord messages into one file"
    docker-compose run --rm "$DOCKER_COMPOSE_SVC" python3 ./scripts/combine-training-data.py
fi

if ! [[ -f"$ENCODING_OUT" ]]; then
    echo "Encoding Discord messages"
    docker-compose run --rm "$DOCKER_COMPOSE_SVC" python3 ./src/encode.py "$COMBINED_TRAINING_DATA_RELATIVE_TO_ROOT_PATH" "$ENCODING_OUT_RELATIVE_TO_ROOT_PATH"
fi
