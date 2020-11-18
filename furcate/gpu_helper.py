def get_gpus(framework):
    if framework == 'tf':
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
    else:
        raise TypeError("The supplied framework '{}' is not supported.".format(framework))

    return gpus