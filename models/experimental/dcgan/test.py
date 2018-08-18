import tensorflow as tf
with tf.gfile.GFile("gs://ptosis-test/data/img/*.jpg", "r") as f:
        content = f.readlines()