#!/usr/bin/env bash
declare -r DOWNLOADER_IMAGE="tyrrrz/discordchatexporter"
declare -r OUTPUT_DIR="discord-messages/"

declare -a MISSING_ENVS=()
if [[ -z "$DISCORD_TOKEN" ]]; then
    MISSING_ENVS+=("DISCORD_TOKEN")
fi

if [[ -z "$DISCORD_GUILD" ]]; then
    MISSING_ENVS+=("DISCORD_GUILD")
fi

if (( ${#MISSING_ENVS[@]} > 0)); then
    echo "Error: ${MISSING_ENVS[*]} env var(s) must be set" >&2
    exit 1
fi

docker run -it --rm -v "$PWD:/out" "$DOWNLOADER_IMAGE" exportguild --guild "$DISCORD_GUILD" --output "$OUTPUT_DIR" --format "Json" --token "$DISCORD_TOKEN"
