import tensorflow as tf
from PIL import Image
import numpy as np
import os


# Get images and labels into .bin locally

images = []
labels = []


dirs = os.listdir('$/ABSOLUTE-PATH/tpu/models/experimental/dcgan/data/pics/')

# How many images you want to transform
total_img_to_load = 669

for d in dirs[0:total_img]:
    im = ('$/ABSOLUTE-PATH/tpu/models/experimental/dcgan/data/pics/{}'.format(d))
    images.append(im)
    print(images)
    print(len(images))

for l in range(total_img):
    l = 0
    labels.append(l)
    print(labels)
    print(len(labels))


with open("$/ABSOLUTE-PATH/tpu/models/experimental/dcgan/output.bin", "wb") as f:
  for label, img in zip(labels, images): 
    label = np.array(label, dtype=np.uint8)
    f.write(label.tostring())  # Write label.

    im = np.array(Image.open(img), dtype=np.uint8)
    f.write(im[:, :, 0].tostring())  # Write red channel.
    f.write(im[:, :, 1].tostring())  # Write green channel.
    f.write(im[:, :, 2].tostring())  # Write blue channel.

