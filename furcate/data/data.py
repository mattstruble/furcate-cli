def gen_tfrecord_dataset(fork, filepaths, processor, shuffle=False):
    import tensorflow as tf
    dataset = tf.data.TFRecordDataset(filepaths, num_parallel_reads=fork.num_parallel_reads)

    if fork.cache:
        dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(buffer_size=fork.shuffle_buffer_size, seed=fork.seed)

    dataset = dataset.map(processor, num_parallel_calls=fork.num_parallel_calls).batch(fork.batch_size)

    return dataset.prefetch(fork.prefetch)