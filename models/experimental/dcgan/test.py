import tensorflow as tf
# with tf.gfile.GFile("gs://ptosis-test/data/img/", "r") as f:
#         content = f.readlines()
#         print(content)
filename = "gs://ptosis-test/data/img/*.jpg"

with gfile.GFile(filename, "r") as mapping_file:
    lines = mapping_file.readlines()
    mapping = dict([_.split("\t")[0:2] for _ in lines])
    mapping = {k.strip(): v.strip() for k, v in mapping.items()}
return mapping 