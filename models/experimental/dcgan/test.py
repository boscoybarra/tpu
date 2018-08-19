import tensorflow as tf
from PIL import Image
import numpy as np


names = tf.gfile.ListDirectory('gs://ptosis-test/data/img/')



# im = map('gs://ptosis-test/data/img/{}'.format, name)

# colors = ["red", "green", "blue", "purple"]
for name in names:
    im = ('gs://ptosis-test/data/img/{}'.format, name)
    print("HOLA",im)
    # im = (np.array(im))



r = im[:,:,0].flatten()
g = im[:,:,1].flatten()
b = im[:,:,2].flatten()
label = [1]

outfile = "./data"
out = np.array(list(label) + list(r) + list(g) + list(b),np.uint8)
out.tofile("out.bin")
# np.save(outfile, x)