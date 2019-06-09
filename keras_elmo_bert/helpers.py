import tensorflow as tf


def masked_reduce_mean(input, mask):
    """
    Get mean value of tensor values under mask
    :param input: tensor
    :type input: tf.Tensor
    :param mask: mask tensor
    :type mask: tf.Tensor
    :return: mean value
    :rtype: tf.Tensor
    """
    mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
    masked_sum = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1)
    mask_element_count = lambda m: tf.reduce_sum(m, axis=1, keepdims=True)
    masked_reduce_mean = lambda x, m: masked_sum(x, m) / (mask_element_count(m) + 1e-10)
    return masked_reduce_mean(input, mask)
