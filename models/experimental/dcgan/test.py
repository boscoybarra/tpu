import tensorflow as tf
from PIL import Image
import numpy as np


name = tf.gfile.ListDirectory('gs://ptosis-test/data/img/')
print(name)
print(len(name))


# im = map('gs://ptosis-test/data/img/{}'.format, name)

im = tf.gfile.ListDirectory('gs://ptosis-test/data/img/{}'.format, name)
print im
im = (np.array(im))
print("HOLA", im)


r = im[:,:,0].flatten()
g = im[:,:,1].flatten()
b = im[:,:,2].flatten()
label = [1]

outfile = "./data"
out = np.array(list(label) + list(r) + list(g) + list(b),np.uint8)
out.tofile("out.bin")
# np.save(outfile, x)