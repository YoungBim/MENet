import tensorflow as tf
import numpy as np

def weighted_cross_entropy(onehot_labels, logits, class_weights):
    '''
    A quick wrapper to compute weighted cross entropy.

    ------------------
    Technical Details
    ------------------
    The class_weights list can be multiplied by onehot_labels directly because the last dimension
    of onehot_labels is 12 and class_weights (length 12) can broadcast across that dimension, which is what we want.
    Then we collapse the last dimension for the class_weights to get a shape of (batch_size, height, width, 1)
    to get a mask with each pixel's value representing the class_weight.

    This mask can then be that can be broadcasted to the intermediate output of logits
    and onehot_labels when calculating the cross entropy loss.
    ------------------

    INPUTS:
    - onehot_labels(Tensor): the one-hot encoded labels of shape (batch_size, height, width, num_classes)
    - logits(Tensor): the logits output from the model that is of shape (batch_size, height, width, num_classes)
    - class_weights(list): A list where each index is the class label and the value of the index is the class weight.

    OUTPUTS:
    - loss(Tensor): a scalar Tensor that is the weighted cross entropy loss output.
    '''
    weights = onehot_labels * class_weights
    weights = tf.reduce_sum(weights, 3)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits, weights=weights)

    return loss

def compute_smooth_loss(pred_disp):
    def gradient(pred):
        D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        return D_dx, D_dy
    dx, dy = gradient(pred_disp)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)
    return tf.reduce_mean(tf.abs(dx2)) + \
           tf.reduce_mean(tf.abs(dxdy)) + \
           tf.reduce_mean(tf.abs(dydx)) + \
           tf.reduce_mean(tf.abs(dy2))

# Weighted Cross entropy loss
def segmentation_loss_wce(task, pred, anots, num_classes, class_weights):
    # perform one-hot-encoding on the ground truth annotation to get same shape as the logits
    annotations_ohe = tf.one_hot(anots, num_classes, axis=-1, name=task + '_ohe')
    # Actually compute the loss
    loss = weighted_cross_entropy(logits=pred, onehot_labels=annotations_ohe, class_weights=class_weights)
    return loss

# Naive L1 depth loss
def depth_loss_nL1(task, pred, anots, scope = '_L1' ):
    numel = pred.get_shape().as_list()[1] * pred.get_shape().as_list()[2]
    # Get the predicted depth squeezed
    pdpth = tf.reshape(tf.squeeze(pred, axis=3, name=task + scope + '_pred_squeeze'), [-1, numel], name=task + scope + '_pred_flat')
    # Define parts of the GT that are defined
    Defined = tf.reshape(tf.greater(anots, tf.constant(0, dtype=anots.dtype), name=task + scope + '_GT_isDefined'), [-1, numel], name=task + scope + '_GT_isDefined_flat')
    # Compute the absolute value between prediction and GT (full)
    proj_error = tf.abs(pdpth - tf.reshape(tf.cast(anots, tf.float32), [-1, numel]), name=task + scope + '_diff')
    # Mask parts of the result where GT is not defined
    proj_error = tf.boolean_mask(proj_error, Defined, name=task + scope + '_diff_where_isDefined')
    # Take the reduced mean of it
    loss = tf.reduce_mean(proj_error, name=task + scope + '_loss')
    # Normalize the loss so that it resembles the segmentation Loss
    lmbd = tf.constant(500.0, tf.float32)
    loss = tf.divide(loss, lmbd, name=task + scope + '_norm_loss')
    return loss

# Naive L1 depth loss + L2 Regularization Term
def depth_loss_nL1_Reg(task, pred, anots, scope = '_regL1') :
    loss = depth_loss_nL1(task, pred, anots)
    smooth = tf.multiply(compute_smooth_loss(pred), tf.constant(0.001, tf.float32, name=task + scope + '_norm_loss'))
    loss = tf.add(loss, smooth)
    return loss