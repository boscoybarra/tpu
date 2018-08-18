import tensorflow as tf
# with tf.gfile.GFile("gs://ptosis-test/data/img/", "r") as f:
#         content = f.readlines()
#         print(content)
# filename = "gs://ptosis-test/data/img/*.jpg"

def _get_unk_mapping(filename):
  """Reads a file that specifies a mapping from source to target tokens.
  The file must contain lines of the form <source>\t<target>"

  Args:
    filename: path to the mapping file

  Returns:
    A dictionary that maps from source -> target tokens.
  """
  with tf.gfile.ListDirectory(filename, "r") as mapping_file:
    lines = mapping_file.readlines()
    mapping = dict([_.split("\t")[0:2] for _ in lines])
    mapping = {k.strip(): v.strip() for k, v in mapping.items()}
  return lines

print(_get_unk_mapping('gs://ptosis-test/data/img/'))