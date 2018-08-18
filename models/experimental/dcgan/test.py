import tensorflow as tf
with tf.gfile.GFile("gs://ptosis-test/data/img/", "r") as f:
        content = f.readlines()
        print(content)