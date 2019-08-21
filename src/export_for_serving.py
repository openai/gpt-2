#!/usr/bin/env python3

import fire
import time
import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

import model, sample

def export_for_serving(
    model_name='124M',
    seed=None,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    models_dir='models'
):
    """
    Export the model for TF Serving
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))

    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)

        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        export_dir=os.path.join(models_dir, model_name, "export", str(time.time()).split('.')[0])
        if not os.path.isdir(export_dir):
            os.makedirs(export_dir)

        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
        signature = predict_signature_def(inputs={'context': context},
        outputs={'sample': output})

        builder.add_meta_graph_and_variables(sess,
                                     [tf.saved_model.SERVING],
                                     signature_def_map={"predict": signature},
                                     strip_default_attrs=True)
        builder.save()

if __name__ == '__main__':
    fire.Fire(export_for_serving)

