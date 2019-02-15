#!/bin/bash

ID=1kY-qc0uCU3uGPhVGTfryvHr5zWh_B9g7
NAME=117M
DIR=models
GREP_CMD=grep
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
if [ "${OS}" = "darwin" ]; then 
GREP_CMD=pcregrep
fi


curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=${ID}" > /tmp/intermezzo.html
curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | $GREP_CMD -o 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > "${DIR}/${NAME}.tar.gz"
tar -xzvf "${DIR}/${NAME}.tar.gz" -C ${DIR}
rm "${DIR}/${NAME}.tar.gz"

