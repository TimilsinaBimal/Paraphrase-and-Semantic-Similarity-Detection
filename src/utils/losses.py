import tensorflow as tf
from tensorflow.keras import backend as K


def identity_loss(y_true, y_pred):
    return K.mean(y_pred)


def triplet_loss(y_true, y_pred, alpha=0.25):
    anchor = y_pred[:, :128]
    positive = y_pred[:, 128:128 * 2]
    negative = y_pred[:, -128:]
    pos_dist = K.sum(K.square(anchor - positive), axis=1)
    neg_dist = K.sum(K.square(anchor - negative), axis=1)
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(basic_loss, 0.0)
    return loss


def cos_distance(vectors):
    y_true, y_pred = vectors

    def l2_normalize(x, axis):
        norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
        return K.sign(x) * K.maximum(K.abs(x), K.epsilon()) / K.maximum(norm, K.epsilon())
    y_true = l2_normalize(y_true, axis=1)
    y_pred = l2_normalize(y_pred, axis=1)
    return K.mean(1 - y_true * y_pred, axis=1)


def contrastive_loss(y, preds, margin=1):
    y = tf.cast(y, preds.dtype)
    squaredPreds = K.square(preds)
    squaredMargin = K.square(K.maximum(margin - preds, 0))
    loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
    return loss


def euclidean_distance(vectors):
    (featsA, featsB) = vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))


def cos_distance(vectors):
    y_true, y_pred = vectors

    def l2_normalize(x, axis):
        norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
        return K.sign(x) * K.maximum(K.abs(x), K.epsilon()) / K.maximum(norm, K.epsilon())
    y_true = l2_normalize(y_true, axis=-1)
    y_pred = l2_normalize(y_pred, axis=-1)
    return K.mean(y_true * y_pred, axis=-1)


def TripletLoss(margin=0.25):
    def triplet(y_true, y_pred):
        batch_size = tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
        v1, v2 = y_pred[:, :128], y_pred[:, -128:]
        scores = K.dot(v1, K.transpose(v2))
        positive = tf.linalg.diag_part(scores)
        negative_without_positive = scores - 2 * tf.eye(batch_size)

        closest_negative = tf.reduce_max(negative_without_positive, axis=1)

        negative_zero_on_duplicate = scores * (1.0 - tf.eye(batch_size))

        mean_negative = K.sum(negative_zero_on_duplicate,
                              axis=1) / (batch_size - 1)

        triplet_loss1 = K.maximum(0.0, margin - positive + closest_negative)

        triplet_loss2 = K.maximum(0.0, margin - positive + mean_negative)

        triplet_loss = K.mean(triplet_loss1 + triplet_loss2)

        return triplet_loss
    return triplet
