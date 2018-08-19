import tensorflow as tf
from PIL import Image
import numpy as np

urls= []
names = tf.gfile.ListDirectory('gs://ptosis-test/data/img/')

for name in names:
    im = ('gs://ptosis-test/data/img/{}'.format(name))
    urls.append(im)
    print(urls)




# im = map('gs://ptosis-test/data/img/{}'.format, name)

# colors = ["red", "green", "blue", "purple"]

# with open(output_filename, "wb") as f:
#   for label, img in zip(labels, images): 
#     label = np.array(label, dtype=np.uint8)
#     f.write(label.tostring())  # Write label.

#     im = np.array(Image.open(img), dtype=np.uint8)
#     f.write(im[:, :, 0].tostring())  # Write red channel.
#     f.write(im[:, :, 1].tostring())  # Write green channel.
#     f.write(im[:, :, 2].tostring())  # Write blue channel.

# for name in names:
    
#     im = ('gs://ptosis-test/data/img/{}'.format(name))
#     print(im)

#     # IMport from dir instead of gs:// ??
#     with tf.gfile.GFile(im, "r") as f:
#         content = f.read()
#         content = np.array(content)
#         print(content)
#         # im = tf.gfile.GFile.read(n=-1)
#         r = content[:,:,0].flatten()
#         g = content[:,:,1].flatten()
#         b = content[:,:,2].flatten()
#         label = [1]

#         outfile = "./data"
#         out = np.array(list(label) + list(r) + list(g) + list(b),np.uint8)
#         out.tofile("out.bin")
#         np.save(outfile, out)

