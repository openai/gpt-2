import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
import sys
from encoder import get_encoder
import model

# =====================
# Configuration
# =====================
checkpoint_dir = "checkpoints_from_scratch"
seq_len = 1024
gen_length = 200
temperature = 0.5
top_k = 40

# Optional: pass a prompt as command line argument
prompt_text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""

# =====================
# Load encoder
# =====================
enc = get_encoder("124M", "models")

# =====================
# Build model
# =====================
hparams = model.default_hparams().copy()
hparams.n_ctx = seq_len
hparams.n_embd = 768
hparams.n_layer = 12
hparams.n_head = 12
hparams.n_vocab = 50257

X = tf.placeholder(tf.int32, [1, None], name="X")
outputs = model.model(hparams=hparams, X=X)
logits = outputs['logits']

saver = tf.train.Saver()

def top_k_filter(logits, k):
    values, _ = tf.nn.top_k(logits, k=k)
    min_val = values[:, -1:]
    return tf.where(logits < min_val, tf.fill(tf.shape(logits), -1e10), logits)

# =====================
# Generate
# =====================
with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if ckpt is None:
        print(f"No checkpoint found in {checkpoint_dir}")
        sys.exit(1)
    print(f"Loading checkpoint: {ckpt}")
    saver.restore(sess, ckpt)

    while(True):
        prompt_text = input("\n Input Starter C Code  >>> ")
        if prompt_text == "exit":
            exit()

        # Encode prompt or start with endoftext token
        if prompt_text:
            tokens = enc.encode(prompt_text)
        else:
            tokens = [enc.encoder['<|endoftext|>']]

        print("=" * 50)
        if prompt_text:
            print(prompt_text, end="", flush=True)

        for _ in range(gen_length):
            # Keep within context window
            input_tokens = tokens[-seq_len:]
            feed = np.array([input_tokens], dtype=np.int32)
            logits_val = sess.run(logits, feed_dict={X: feed})

            # Get logits for next token
            next_logits = logits_val[0, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                top_values = np.sort(next_logits)[::-1][:top_k]
                min_val = top_values[-1]
                next_logits[next_logits < min_val] = -1e10

            # Sample (numerically stable softmax)
            next_logits = next_logits - np.max(next_logits)
            probs = np.exp(next_logits) / np.sum(np.exp(next_logits))
            next_token = np.random.choice(len(probs), p=probs)

            tokens.append(int(next_token))
            print(enc.decode([int(next_token)]), end="", flush=True)

        print("\n" + "=" * 50)
    print("Goodbye")
