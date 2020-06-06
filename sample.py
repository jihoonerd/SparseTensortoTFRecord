import pandas as pd
import tensorflow as tf
import numpy as np
from sys import getsizeof
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Lambda
from tqdm import tqdm

raw_path = "/home/jihoon/melon-playlist-continuation/data/raw/"

TOTAL_SONGS_NUM = 707_989

df = pd.read_json(raw_path + "train.json")
df.head()

playlists = df["songs"].values

# use tqdm
sparse_mtxs = []
for playlist in tqdm(playlists[:1200]):
    indices = []
    for song_id in playlist:
        indices.append([0, song_id])
    values = np.ones(len(indices), dtype=np.int8)
    dense_shape = [1, TOTAL_SONGS_NUM]
    
    sparse_mtxs.append(tf.sparse.reorder(tf.sparse.SparseTensor(indices, values, dense_shape)))


serialized_sparse_tensors = [tf.io.serialize_sparse(st).numpy() for st in sparse_mtxs]

with tf.io.TFRecordWriter('sparse_example.tfrecord') as tfwriter:
    for sst in serialized_sparse_tensors:
        sparse_example = tf.train.Example(features = 
                     tf.train.Features(feature=
                         {'sparse_tensor': 
                               tf.train.Feature(bytes_list=tf.train.BytesList(value=sst))
                         }))
        # Append each example into tfrecord
        tfwriter.write(sparse_example.SerializeToString())

def parse_fn(data_element):
    features = {'sparse_tensor': tf.io.FixedLenFeature([3], tf.string)}
    parsed = tf.io.parse_single_example(data_element, features=features)

    # tf.io.deserialize_many_sparse() requires the dimensions to be [N,3] so we add one dimension with expand_dims
    parsed['sparse_tensor'] = tf.expand_dims(parsed['sparse_tensor'], axis=0)
    # deserialize sparse tensor
    parsed['sparse_tensor'] = tf.io.deserialize_many_sparse(parsed['sparse_tensor'], dtype=tf.int8)
    # convert from sparse to dense
    parsed['sparse_tensor'] = tf.sparse.to_dense(parsed['sparse_tensor'])
    # remove extra dimenson [1, 3] -> [3]
    parsed['sparse_tensor'] = tf.squeeze(parsed['sparse_tensor'])
    return parsed

# Read from TFRecord
dataset = tf.data.TFRecordDataset(['sparse_example.tfrecord'])
dataset = dataset.map(parse_fn)

for minibatch in dataset.batch(100):
    print(minibatch['sparse_tensor'].numpy().sum())