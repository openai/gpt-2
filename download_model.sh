#!/bin/bash
# if you have a proxy blocked you using download_model.py to get models, try to use this curl shell. 
filenames=(checkpoint encoder.json hparams.json model.ckpt.data-00000-of-00001 model.ckpt.index model.ckpt.meta vocab.bpe)
for f in "{filenames[@]}"
do
    echo $f
    curl -fsSL https://storage.googleapis.com/gpt-2/models/117M/$f -o ./models/117M/$f
done
