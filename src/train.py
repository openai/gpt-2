import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
from encoder import get_encoder
import model

# =====================
# Configuration
# =====================
text_file = os.path.join(os.path.dirname(__file__), "cprogramming")  # your training text
batch_size = 2 
seq_len = 1024
num_steps = 100000
save_every = 10000
learning_rate = 1e-4
checkpoint_dir = "checkpoints_from_scratch"

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# =====================
# Load and encode text
# =====================
enc = get_encoder("124M", "models")

with open(text_file, "r", encoding="utf-8") as f:
    text = f.read()

tokens = np.array(enc.encode(text), dtype=np.int32)
print(f"Total tokens in dataset: {len(tokens)}")

# =====================
# Build GPT-2 model
# =====================
hparams = model.default_hparams().copy()
hparams.n_ctx = seq_len
hparams.n_embd = 768
hparams.n_layer = 12
hparams.n_head = 12
hparams.n_vocab = 50257

X = tf.placeholder(tf.int32, [None, None], name="X")

outputs = model.model(hparams=hparams, X=X)
logits = outputs['logits']

# Shift logits and labels for next-token prediction
labels = X[:, 1:]
logits = logits[:, :-1, :]

# Loss
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
loss = tf.reduce_mean(loss)

# Optimizer with gradient clipping
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9)
gradients, variables = zip(*optimizer.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
train_op = optimizer.apply_gradients(zip(gradients, variables))

# Saver for checkpoints
saver = tf.train.Saver(max_to_keep=5)

# =====================
# Batch generator
# =====================
def get_batch(tokens, batch_size, seq_len):
    idx = np.random.randint(0, len(tokens) - seq_len - 1, size=batch_size)
    batch = [tokens[i:i + seq_len + 1] for i in idx]
    return np.array(batch, dtype=np.int32)

# =====================
# Training loop
# =====================
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1, num_steps + 1):
        batch = get_batch(tokens, batch_size, seq_len)
        loss_val, _ = sess.run([loss, train_op], feed_dict={X: batch})

        if step % 100 == 0:
            print(f"Step {step}/{num_steps} | Loss: {loss_val:.4f}")

        if step % save_every == 0:
            saver.save(sess, os.path.join(checkpoint_dir, "gpt2"), global_step=step)
            print(f"Checkpoint saved at step {step}")

