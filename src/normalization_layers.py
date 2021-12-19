import numpy as np
import tensorflow as tf

"""
    order: nl (no norm), l1 (L1 norm), l2 (L2 norm), li (L-infinite norm), ll (L-learnable norm)
    radius: nr (no radius), ur (unit radius), lr (learnable radius)
"""

# add epsilon offset to prevent divide by zero
epsilon = 1e-19


# l1, ur
class L1NormUnitRadiusLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(L1NormUnitRadiusLayer, self).__init__()

    def call(self, input_tensor):
        norm = tf.norm(input_tensor, ord=1, axis=1)
        return input_tensor / (tf.reshape(norm, [-1, 1]) + epsilon)


# l2, ur
class L2NormUnitRadiusLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(L2NormUnitRadiusLayer, self).__init__()

    def call(self, input_tensor):
        norm = tf.norm(input_tensor, ord=2, axis=1)
        return input_tensor / (tf.reshape(norm, [-1, 1]) + epsilon)


# li, ur
class LinfNormUnitRadiusLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(LinfNormUnitRadiusLayer, self).__init__()

    def call(self, input_tensor):
        norm = tf.norm(input_tensor, ord=np.inf, axis=1)
        return input_tensor / (tf.reshape(norm, [-1, 1]) + epsilon)


# ll, ur
class LlearnNormUnitRadiusLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(LlearnNormUnitRadiusLayer, self).__init__()
        self.order = tf.Variable(2., trainable=True, name='norm_order')

    def call(self, input_tensor):
        norm_exp = tf.math.pow(tf.reduce_sum(tf.math.pow(input_tensor, self.order)), tf.math.divide(1, self.order))
        return tf.math.divide_no_nan(input_tensor, norm_exp)


# l2, lr
class L2NormLearnRadiusLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(L2NormLearnRadiusLayer, self).__init__()
        self.radius = tf.Variable(1., trainable=True, name='norm_radius')

    def call(self, input_tensor):
        norm = tf.norm(input_tensor, ord=2, axis=1)
        return tf.math.scalar_mul(self.radius, input_tensor / (tf.reshape(norm, [-1, 1]) + epsilon))


# ll, lr
class LlearnNormLearnRadiusLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(LlearnNormLearnRadiusLayer, self).__init__()
        self.order = tf.Variable(2., trainable=True, name='norm_order')
        self.radius = tf.Variable(1., trainable=True, name='norm_radius')

    def call(self, input_tensor):
        norm_exp = tf.math.pow(tf.reduce_sum(tf.math.pow(input_tensor, self.order)), tf.math.divide(1, self.order))
        return tf.math.scalar_mul(self.radius, tf.math.divide_no_nan(input_tensor, norm_exp))


def get_normalization_layer(order='l2', radius='lr'):
    if order == 'l1' and radius == 'ur':
        return L1NormUnitRadiusLayer()
    if order == 'l2' and radius == 'ur':
        return L2NormUnitRadiusLayer()
    if order == 'li' and radius == 'ur':
        return LinfNormUnitRadiusLayer()
    if order == 'll' and radius == 'ur':
        return LlearnNormUnitRadiusLayer()
    if order == 'l2' and radius == 'lr':
        return L2NormLearnRadiusLayer()
    if order == 'll' and radius == 'lr':
        return LlearnNormLearnRadiusLayer()
