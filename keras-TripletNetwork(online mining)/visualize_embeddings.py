'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras

from triplet_losses import batch_all_triplet_loss, batch_hard_triplet_loss
from triplet_metrics import triplet_accuracy, mean_norm
from model import build_mnist_model
from dataset.mnist import load_mnist

import os
import pathlib
import shutil

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

is_mnist = False

batch_size = 128
num_classes = 10
epochs = 20

model = build_mnist_model()
print(model.summary())

model.compile(
    loss=batch_all_triplet_loss,
    optimizer='adam',
    metrics=[triplet_accuracy, mean_norm]
)

model.load_weights('./model.h5')

x_train, y_train, x_test, y_test = load_mnist()

embeddings = model.predict(x_test)
print(embeddings.shape)

tf.logging.info("Embeddings shape: {}".format(embeddings.shape))

# Visualize test embeddings
embedding_var = tf.Variable(embeddings, name='embedding')

eval_dir = "./graph"
summary_writer = tf.summary.FileWriter(eval_dir)

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name

# Specify where you find the sprite (we will create this later)
# Copy the embedding sprite image to the eval directory
if is_mnist:
    shutil.copy2("mnist_10k_sprite.png", eval_dir)
    embedding.sprite.image_path = pathlib.Path("mnist_10k_sprite.png").name
    embedding.sprite.single_image_dim.extend([28, 28])

labels = y_test
# Specify where you find the metadata
# Save the metadata file needed for Tensorboard projector
metadata_filename = "metadata.tsv"
with open(os.path.join(eval_dir, metadata_filename), 'w') as f:
    for i in range(len(labels)):
        c = labels[i]
        f.write('{}\n'.format(c))
embedding.metadata_path = metadata_filename

# Say that you want to visualise the embeddings
projector.visualize_embeddings(summary_writer, config)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(embedding_var.initializer)
    saver.save(sess, os.path.join(eval_dir, "embeddings.ckpt"))
