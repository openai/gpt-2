#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import requests
import cursor


#Added a progress bar since curl has it, source is below
#https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a
#Its slightly modified tho, see comments
# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    cursor.hide()
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
	#Added for files smaller than the buffer
    if iteration>total:
        total=iteration
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    sys.stdout.write("\r%s |%s| %s%s %s" % (prefix, bar, percents, "%", suffix))
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()
        cursor.show()

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
		iterator = 0
		total = int(r.headers["content-length"])
		#went for 1k for chunk_size. Motivation -> Ethernet packet size is around 1500 bytes.
		for chunk in r.iter_content(chunk_size=1000):
			currentFile.write(chunk)
			iterator+=1000
			print_progress(iterator,total,"Fetching "+filename,"",1,10)
			
