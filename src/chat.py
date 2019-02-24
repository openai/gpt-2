#!/usr/bin/env python3

import fire
import json
import os
import sys
import numpy as np
import tensorflow as tf

import model, sample, encoder

def interact_model(
    model_name='117M',
    seed=None,
    length=20,
    temperature=1,
    top_k=0,
    conversation="""
you: hi
her: hey
you: i'm a human
her: i'm a robot
you: you ready?
her: yes :)
you: ok let's start chatting
her: sure, what do you want to talk about?"""
):

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        np.random.seed(seed)
        tf.set_random_seed(seed)
        context = tf.placeholder(tf.int32, [1, None])
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=1,
            temperature=temperature, top_k=top_k
        )

        print(conversation)

        while True: 
            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
            saver.restore(sess, ckpt)
            message = None
            while not message:
                message = input("you: ")
            conversation = conversation + "\nyou: " + message
            conversation = conversation + "\nher: "
            sys.stdout.write("her: ")
            sys.stdout.flush()

            #sys.stderr.write("************************"+conversation+"***********************")
            #sys.stderr.flush()
            
            encoded_conversation = enc.encode(conversation)
            #print(len(encoded_conversation))
            result = sess.run(output, feed_dict={
                context: [encoded_conversation]
            })[:, len(encoded_conversation):]
            text = enc.decode(result[0])
            
            #sys.stderr.write("=============="+text+"=================")
            #sys.stderr.flush()

            splits = text.split('\n')
            #line = splits[1] if len(splits)>1 else splits[0]
            #parts = line.split(': ')
            #reply = parts[1] if len(parts)>1 else parts[0]
            reply = splits[0]
            sys.stdout.write(reply+'\n')
            sys.stdout.flush()
            conversation = conversation + reply

if __name__ == '__main__':
    fire.Fire(interact_model)

