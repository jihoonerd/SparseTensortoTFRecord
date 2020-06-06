import os

import numpy as np
import tensorflow as tf

from converter import st2tfr, tfr2tfd

def test_sparse_tensor_to_tfrecord(remove_output=True):

    # Generate very sparse matrix
    num_samples = 101
    num_cols = 1000
    
    sparse_mtxs = []
    
    for _ in range(num_samples):
        indices = []
        col_id = list(set(np.random.randint(num_cols, size=(num_cols,))))
        for col_id in sorted(col_id):
            # IMPORTANT: it needs to be assumed that each sample (row) is independent, which means they all have row indices as 0.
            indices.append([0, col_id])
        
        values = np.ones(len(indices), dtype=np.int8)
        # Set dense shape as (1, NUM COLS), since main purpose of it is converting out-of-memory data to TFRecord and loading from it.
        dense_shape = [1, num_cols]
        
        sparse_mtxs.append(tf.sparse.SparseTensor(indices, values, dense_shape))
    
    # Make tf.sparse.SparseTensor
    assert len(sparse_mtxs) == num_samples

    st2tfr(sparse_tensors=sparse_mtxs, tfr_filename="test.tfrecord")
    assert os.path.exists("test.tfrecord")

    if remove_output is True:
        os.remove("test.tfrecord")


def test_tfrecord_to_tfrecord_dataset():

    test_sparse_tensor_to_tfrecord(remove_output=False)
    assert os.path.exists("test.tfrecord")
    dataset = tfr2tfd("test.tfrecord")

    batch_count = 0
    for minibatch in dataset.batch(10):
        print(minibatch['sparse_tensor'].numpy().sum())
        batch_count +=1
    
    assert batch_count == 11
    os.remove("test.tfrecord")
