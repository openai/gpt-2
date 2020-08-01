import json
import os
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfcV1
import model, sample, encoder

def conditional_model(
    model_name='345M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=40,
    top_p=0,
    models_dir='models',
    sentences=None,
    ):
    """
    Run the model on multilple sentences and return a dict.
    :model_name : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=40 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :top_p=0.0 : Float value controlling diversity. Implements nucleus sampling,
     overriding top_k if set to a value > 0. A good setting is 0.9.
    :sentences : List of strings or string. Model returns an answer or a continuation
     to that string. If list of strings the model return a dictionary of sentences and their
     respective model replies.
    """
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0
    
    if sentences == None:
        raise ValueError('Sentences cannot be None')

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tfcV1.Session(graph=tf.Graph()) as sess:
        context = tfcV1.placeholder(tf.int32, [batch_size, None])
        tfcV1.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)
        listy = []
        n = 0
        
        if isinstance(sentences, list):
            for i in sentences:
                context_tokens = enc.encode(i)
                for _ in range(nsamples // batch_size):
                    out = sess.run(output, feed_dict={
                        context: [context_tokens for _ in range(batch_size)]
                    })[:, len(context_tokens):]
                text = i + enc.decode(out[0])
                listy.append(text)
                n += 1
                print(n)
            return dict(zip(sentences,listy))
        else:
            context_tokens = enc.encode(sentences)
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
            text = sentences + enc.decode(out[0])
            
            return {sentences: text}
