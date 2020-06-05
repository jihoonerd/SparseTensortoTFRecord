import os

import numpy as np
import tensorflow as tf

from st2tfr import st2tfr


def test_sparse_tensor_to_tfrecord():

    if os.path.exists("test.proto"):
        os.remove("test.proto")
    # Generate very sparse matrix
    num_samples = 100
    num_cols = 1000
    
    sparse_mtxs = []
    
    for row_ind in range(num_samples):
        indices = []
        num_col_fill = np.random.randint(num_cols)
        for _ in range(num_col_fill):
            indices.append((row_ind, np.random.randint(num_cols)))
        
        values = np.ones(len(indices), dtype=np.int8)
        # Set dense shape as (1, NUM COLS), since main purpose of it is converting out-of-memory data to TFRecord and loading from it.
        dense_shape = (1, num_cols)
        
        sparse_tensor = tf.sparse.SparseTensor(indices, values, dense_shape)
        sparse_mtxs.append(sparse_tensor)
    
    # Make tf.sparse.SparseTensor
    assert len (sparse_mtxs) == num_samples

    st2tfr(sparse_tensors=sparse_mtxs, tfr_filename="test.proto")
    assert os.path.exists("test.proto")
    os.remove("test.proto")
