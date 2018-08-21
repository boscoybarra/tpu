import tensorflow as tf
from PIL import Image
import numpy as np
import os
from scipy.sparse import csr_matrix



images = []
# images2 = csr_matrix(images)
labels = []
# labels2 = csr_matrix(images)


dirs = os.listdir('/Users/jb/Documents/dl/tpu/models/experimental/dcgan/data/pics/')

# How many images you want to transform
# total_img_to_load = 669

for d in dirs[0:669]:
    # im = ('/Users/jb/Documents/dl/tpu/models/experimental/dcgan/data/pics/{}'.format(d))
    im = ('gs://ptosis-test/data/{}'.format(d))
    images.append(im)
    print(len(images))
    
print(images)

for l in range(669):
    l += 0
    labels.append(l)
    print(labels)
    print(len(labels))


convert_to(images, labels, 'output')

