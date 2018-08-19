import tensorflow as tf
from PIL import Image
import numpy as np


names = tf.gfile.ListDirectory('gs://ptosis-test/data/img/')

# im = map('gs://ptosis-test/data/img/{}'.format, name)

# colors = ["red", "green", "blue", "purple"]
for name in names:
    
    im = ('gs://ptosis-test/data/img/{}'.format(name))
    print(im)

    # IMport from dir instead of gs:// ??
    im = np.array(tf.gfile.GFile(tf.gfile.FastGFile(im, 'r')).read(n=-1), dtype=np.uint8)
    r = im[:,:,0].flatten()
    g = im[:,:,1].flatten()
    b = im[:,:,2].flatten()
    label = [1]

    outfile = "./data"
    out = np.array(list(label) + list(r) + list(g) + list(b),np.uint8)
    out.tofile("out.bin")
    np.save(outfile, out)


# The CIFAR-10 binary format represents each example as a fixed-length record with the following format:

# 1-byte label.
# 1 byte per pixel for the red channel of the image.
# 1 byte per pixel for the green channel of the image.
# 1 byte per pixel for the blue channel of the image.
# Assuming you have a list of image filenames called images, and a list of integers (less than 256) called labels corresponding to their labels, the following code would write a single file containing these images in CIFAR-10 format:

# with open(output_filename, "wb") as f:
#   for label, img in zip(labels, images): 
#     label = np.array(label, dtype=np.uint8)
#     f.write(label.tostring())  # Write label.

#     im = np.array(Image.open(img), dtype=np.uint8)
#     f.write(im[:, :, 0].tostring())  # Write red channel.
#     f.write(im[:, :, 1].tostring())  # Write green channel.
#     f.write(im[:, :, 2].tostring())  # Write blue channel.