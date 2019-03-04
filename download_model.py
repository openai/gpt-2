#!/usr/bin/env python
import os
import sys
import requests
from tqdm import tqdm

if len(sys.argv)!=2:
    print('You must enter the model name as a parameter, e.g.: download_model.py 117M')
    sys.exit(1)
model = sys.argv[1]
#Create directory if it does not exist already, then do nothing
if not os.path.exists('models/'+model):
    os.makedirs('models/'+model)
#download all the files
for filename in ['checkpoint','encoder.json','hparams.json','model.ckpt.data-00000-of-00001', 'model.ckpt.index', 'model.ckpt.meta', 'vocab.bpe']:
    r = requests.get("https://storage.googleapis.com/gpt-2/models/"+model+"/"+filename,stream=True)
    #wb flag required for windows
    with open('models/'+model+'/'+filename,'wb') as currentFile:
        fileSize = int(r.headers["content-length"])
        with tqdm(ncols=100,desc="Fetching "+filename,total=fileSize,unit_scale=True) as pbar:
            #went for 1k for chunk_size. Motivation -> Ethernet packet size is around 1500 bytes.
            for chunk in r.iter_content(chunk_size=1000):
                currentFile.write(chunk)
                pbar.update(1000)
