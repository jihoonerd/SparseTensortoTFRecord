from typing import List
from tqdm import tqdm

import tensorflow as tf
from tensorflow.sparse import SparseTensor


def st2tfr(sparse_tensors: List[SparseTensor], tfr_filename: str) -> None:
    """Converts a list of SparseTensor into TFRecord

    Args:
        sparse_tensors (List[SparseTensor]): LIST of tf.sparse.SparseTensor. Each SparseTensor should have shape as (1, num_cols)
        tfr_filename (str): filename to be saved
    """
    serialized_sparse_tensors = []
    print("Serializing arrays...")
    for sparse_tensor in tqdm(sparse_tensors):
        # Make sure the indices are ordered
        reordered = tf.sparse.reorder(sparse_tensor)
        # Serialize sparse tensor into a numpy byte(str) (https://www.tensorflow.org/api_docs/python/tf/io/serialize_sparse)
        # A 3-vector (1-D Tensor), with each column representing the serialized SparseTensor's indices, values, and shape (respectively).
        serialized_sparse_tensors.append(tf.io.serialize_sparse(reordered).numpy())

    print("Writing TFRecord...")
    with tf.io.TFRecordWriter(tfr_filename) as writer:
        for serialized in serialized_sparse_tensors:
            sparse_example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "sparse_tensor": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=serialized)
                        )
                    }
                )
            )
            writer.write(sparse_example.SerializeToString())
