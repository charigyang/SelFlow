import numpy as np
import tensorflow as tf

def flow_error_avg(flow_1, flow_2, mask):
    """Evaluates the average endpoint error between flow batches."""
    with tf.variable_scope('flow_error_avg'):
        diff = euclidean(flow_1 - flow_2) * mask
        error = tf.reduce_sum(diff) / tf.reduce_sum(mask)
        return tf.convert_to_tensor(error)


def outlier_ratio(gt_flow, flow, mask, threshold=3.0, relative=0.05):
    diff = euclidean(gt_flow - flow) * mask
    if relative is not None:
        threshold = tf.maximum(threshold, euclidean(gt_flow) * relative)
        outliers = tf.cast(tf.greater_equal(diff, threshold), tf.float32)
    else:
        outliers = tf.cast(tf.greater_equal(diff, threshold), tf.float32)
    ratio = tf.reduce_sum(outliers) / tf.reduce_sum(mask)
    return ratio


def outlier_pct(gt_flow, flow, mask, threshold=3.0, relative=0.05):
    frac = outlier_ratio(gt_flow, flow, mask, threshold, relative) * 100
    return tf.convert_to_tensor(frac)

def merge_dictionaries(dict1, dict2):
    merged_dictionary = {}

    for key in dict1:
        if key in dict2:
            new_value = dict1[key] + dict2[key]
        else:
            new_value = dict1[key]

        merged_dictionary[key] = new_value

    for key in dict2:
        if key not in merged_dictionary:
            merged_dictionary[key] = dict2[key]

    return merged_dictionary

def euclidean(t):
    return tf.sqrt(tf.reduce_sum(t ** 2, [3], keepdims=True))