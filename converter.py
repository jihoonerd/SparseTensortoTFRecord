from typing import List
from tqdm import tqdm

import tensorflow as tf
from tensorflow.sparse import SparseTensor
from tensorflow.data import TFRecordDataset


def st2tfr(sparse_tensors: List[SparseTensor], tfr_filename: str) -> None:
    """Converts a list of SparseTensor into TFRecord
    Args:
        sparse_tensors (List[SparseTensor]): LIST of tf.sparse.SparseTensor. Each SparseTensor should have shape as (1, num_cols)
        tfr_filename (str): filename to be saved
    """
    serialized_sparse_tensors = [tf.io.serialize_sparse(st).numpy() for st in sparse_tensors]
    print("Writing TFRecord...")
    with tf.io.TFRecordWriter(tfr_filename) as tfrwriter:
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
            tfrwriter.write(sparse_example.SerializeToString())



def _parser_st(sample):
    """Parser function for tfr2tfd.

    Args:
        data ([type]): Single data point
    """

    features = {"sparse_tensor": tf.io.FixedLenFeature([3], tf.string)}
    parsed = tf.io.parse_single_example(sample, features=features)
    # tf.io.deserialize_many_tensors takes 2-D Tensor of type string of shape [N, 3]
    parsed['sparse_tensor'] = tf.expand_dims(parsed['sparse_tensor'], axis=0)
    # tf.io.deserialize_many_tensors return: A SparseTensor representing the deserialized SparseTensors, concatenated along the SparseTensors' first dimension.
    parsed['sparse_tensor'] = tf.io.deserialize_many_sparse(parsed['sparse_tensor'], dtype=tf.int8)
    # convert tensor into dense
    parsed['sparse_tensor'] = tf.sparse.to_dense(parsed['sparse_tensor'])
    # Dimension from (1, 3) -> (3)
    parsed['sparse_tensor'] = tf.squeeze(parsed['sparse_tensor'])
    return parsed



def tfr2tfd(tfr_file_name: str) -> TFRecordDataset:
    """Generate TFRecordDataset from given protocol buffer file

    Args:
        tfr_file_name (str): filepath to .proto

    Returns:
        TFRecordDataset: TFRecordDataset
    """

    # Read file from protocol buffer
    dataset = tf.data.TFRecordDataset([tfr_file_name])
    dataset = dataset.map(_parser_st)
    return dataset

