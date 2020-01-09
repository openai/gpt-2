#!/usr/bin/env python3

import json
import os
import numpy as np
import tensorflow as tf
import model, sample, encoder
from flask import Flask, request

app = Flask(__name__)

model_name = '124M'
seed = None
nsamples = 1
batch_size = 1
length = None
temperature = 1
top_k = 0
top_p = 1
models_dir = 'models'

models_dir = os.path.expanduser(os.path.expandvars(models_dir))
if batch_size is None:
    batch_size = 1
assert nsamples % batch_size == 0

enc = encoder.get_encoder(model_name, models_dir)
hparams = model.default_hparams()
with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
    hparams.override_from_dict(json.load(f))

if length is None:
    length = hparams.n_ctx // 2
elif length > hparams.n_ctx:
    raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)


class Serving:
    def __init__(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        self.output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=self.context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(self.sess, ckpt)

    def model_run(self, raw_text):
        context_tokens = enc.encode(raw_text)
        generated = 0
        res = []
        for _ in range(nsamples // batch_size):
            out = self.sess.run(self.output, feed_dict={
                self.context: [context_tokens for _ in range(batch_size)]
            })[:, len(context_tokens):]
            for i in range(batch_size):
                generated += 1
                text = enc.decode(out[i])
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                print(text)
                res.append("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                res.append(text)

        print("=" * 80)
        res.append("=" * 80)
        return "<br/>".join(res)


@app.route('/api', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        raw_text = request.get_json()['text']
    else:
        raw_text = request.args.get("text")
    print("Context: %s" % raw_text)
    texts = serve.model_run(raw_text)
    return texts


if __name__ == '__main__':
    serve = Serving()
    app.run(host='0.0.0.0', port='5000', debug=True)
