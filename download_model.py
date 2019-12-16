import os
import sys
import requests
from tqdm import tqdm
import re
import urllib.request

if len(sys.argv) != 2:
    print('You must enter the model name as a parameter, e.g.: download_model.py 124M')
    print('You can also list the available models using download_model.py list')
    sys.exit(1)

model = sys.argv[1]

url_storage = "https://storage.googleapis.com/gpt-2/"

if model == 'list':
     regex = r"models\/(\d*M)\/"
     model_bucket = urllib.request.urlopen(url_storage)
     model_bucket_content = model_bucket.read()

     available_models = dict()
     matches = re.finditer(regex, str(model_bucket_content), re.MULTILINE)
     available_models = list(dict.fromkeys([match.group(1) for match in matches]))
     available_models.sort(key=lambda x:int(x[:-1]))

     print(available_models)
    
else:
    subdir = os.path.join('models', model)
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    subdir = subdir.replace('\\','/') # needed for Windows

    for filename in ['checkpoint','encoder.json','hparams.json','model.ckpt.data-00000-of-00001', 'model.ckpt.index', 'model.ckpt.meta', 'vocab.bpe']:

        r = requests.get(url_storage + subdir + "/" + filename, stream=True)

        with open(os.path.join(subdir, filename), 'wb') as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)
