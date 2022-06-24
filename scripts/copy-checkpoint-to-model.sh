#!/usr/bin/env bash
declare -r PROG_DIR=$(dirname $(realpath "$0"))

declare -r DEFAULT_OPT_BASE_MODEL="124M"

# Helpers
show_help() {
    cat <<EOF
copy-checkpoint-to-model.sh - Copy training checkpoint files into the model definitions directory

USAGE

  copy-checkpoint-to-model.sh -m <model name> -r <checkpoint run name> -i <iteration count>

OPTIONS

  -m <model name>           Name of the new model
  -r <checkpoint run name>  Name of the training run name
  -i <iteration count>      Iteration count from which training weights should be used
  -b <base model>           Name of the base model from where non training data will be copied (Default: '$DEFAULT_OPT_BASE_MODEL')
  -f                        Overwrite a model if it exists

BEHAVIOR

  Makes a directory in models/ named <model name>. Then copies files from checkpoint/<checkpoint run name>/ into models/<model name>/. Ensuring that the copied files have the correct name. The <iteration count> option specifies which files from checkpoint/<checkpoint run name>/ should be copied.

  If a model with <model name> already exists then the script will exit unless -f is provided.

  The <base model> is the name of a sub-directory in models/ from which other miscellaneous files like the hyperparams, vocab, and encoding configuration files will be taken.

EOF
}

# Options
declare OPT_MODEL_NAME
declare OPT_CHECKPOINT_RUN_NAME
declare OPT_ITERATION_COUNT
declare OPT_BASE_MODEL="$DEFAULT_OPT_BASE_MODEL"
declare OPT_FORCE

while getopts "hm:r:i:b:f" opt; do
    case "$opt" in
	   h)
		  show_help
		  exit 0
		  ;;
	   m) OPT_MODEL_NAME="$OPTARG" ;;
	   r) OPT_CHECKPOINT_RUN_NAME="$OPTARG" ;;
	   i) OPT_ITERATION_COUNT="$OPTARG" ;;
	   b) OPT_BASE_MODEL="$OPTARG" ;;
	   f) OPT_FORCE="y" ;;
	   '?')
		  echo "Error: Unknown option" >&2
		  exit 1
		  ;;
    esac
done

declare -a MISSING_ENVS=()
for env_var_name in OPT_MODEL_NAME OPT_CHECKPOINT_RUN_NAME OPT_ITERATION_COUNT; do
    if [[ -z "${!env_var_name}" ]]; then
	   MISSING_ENVS+=("$env_var_name")
    fi
done

if (( ${#MISSING_ENVS[@]} > 0 )); then
    echo "Error: Missing option values ${MISSING_ENVS[@]}, see -h help text for overview" >&2
    exit 1
fi

# Copy files
declare -r MODEL_DIR="$PROG_DIR/../models/$OPT_MODEL_NAME"
declare -r CHECKPOINT_DIR="$PROG_DIR/../checkpoint/$OPT_CHECKPOINT_RUN_NAME"
declare -r BASE_MODEL_DIR="$PROG_DIR/../models/$OPT_BASE_MODEL"

declare -r CHECKPOINT_DATA_FILE="$CHECKPOINT_DIR/model-$OPT_ITERATION_COUNT.data-00000-of-00001"
declare -r CHECKPOINT_INDEX_FILE="$CHECKPOINT_DIR/model-$OPT_ITERATION_COUNT.index"
declare -r CHECKPOINT_META_FILE="$CHECKPOINT_DIR/model-$OPT_ITERATION_COUNT.meta"

declare -r MODEL_DATA_FILE="$MODEL_DIR/model.ckpt.data-00000-of-00001"
declare -r MODEL_INDEX_FILE="$MODEL_DIR/model.ckpt.index"
declare -r MODEL_META_FILE="$MODEL_DIR/model.ckpt.meta"

declare -ra BASE_MODEL_FILES=("$BASE_MODEL_DIR"/{checkpoint,encoder.json,hparams.json,vocab.bpe})

if ! [[ -d "$CHECKPOINT_DIR" ]]; then
    echo "Error: Checkpoint run with name '$OPT_CHECKPOINT_RUN_NAME' cannot be found" >&2
    exit 1
fi

if ! [[ -d "$BASE_MODEL_DIR" ]]; then
    echo "Error: Base model with name '$OPT_BASE_MODEL' could not be found" >&2
    exit 1
fi

if [[ -d "$MODEL_DIR" ]] && [[ -z "$OPT_FORCE" ]]; then
    echo "Error: Model with name '$OPT_MODEL_NAME' already exists, provide -f option to override this model's data"
    exit 1
fi

mkdir -p "$MODEL_DIR"

cp "$CHECKPOINT_DATA_FILE" "$MODEL_DATA_FILE"
cp "$CHECKPOINT_INDEX_FILE" "$MODEL_INDEX_FILE"
cp "$CHECKPOINT_META_FILE" "$MODEL_META_FILE"
cp "${BASE_MODEL_FILES[@]}" "$MODEL_DIR"
