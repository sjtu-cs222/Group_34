
from __future__ import division

import abc
import collections
import math
import sys

import numpy
import tensorflow as tf
import utils

EpsDelta = collections.namedtuple("EpsDelta", ["spent_eps", "spent_delta"])



class MomentsAccountant(object):

  __metaclass__ = abc.ABCMeta

  def __init__(self, total_examples, moment_orders=32):


    assert total_examples > 0
    self._total_examples = total_examples
    self._moment_orders = (moment_orders
                           if isinstance(moment_orders, (list, tuple))
                           else range(1, moment_orders + 1))
    self._max_moment_order = max(self._moment_orders)
    assert self._max_moment_order < 100, "The moment order is too large."
    self._log_moments = [tf.Variable(numpy.float64(0.0),
                                     trainable=False,
                                     name=("log_moments-%d" % moment_order))
                         for moment_order in self._moment_orders]

  @abc.abstractmethod
  def _compute_log_moment(self, sigma, q, moment_order):

    pass

  def accumulate_privacy_spending(self, unused_eps_delta,
                                  sigma, num_examples):

    q = tf.cast(num_examples, tf.float64) * 1.0 / self._total_examples

    moments_accum_ops = []
    for i in range(len(self._log_moments)):
      moment = self._compute_log_moment(sigma, q, self._moment_orders[i])
      moments_accum_ops.append(tf.assign_add(self._log_moments[i], moment))
    return tf.group(*moments_accum_ops)

  def _compute_delta(self, log_moments, eps):

    min_delta = 1.0
    for moment_order, log_moment in log_moments:
      if math.isinf(log_moment) or math.isnan(log_moment):
        sys.stderr.write("The %d-th order is inf or Nan\n" % moment_order)
        continue
      if log_moment < moment_order * eps:
        min_delta = min(min_delta,
                        math.exp(log_moment - moment_order * eps))
    return min_delta

  def _compute_eps(self, log_moments, delta):
    min_eps = float("inf")
    for moment_order, log_moment in log_moments:
      if math.isinf(log_moment) or math.isnan(log_moment):
        sys.stderr.write("The %d-th order is inf or Nan\n" % moment_order)
        continue
      min_eps = min(min_eps, (log_moment - math.log(delta)) / moment_order)
    return min_eps

  def get_privacy_spent(self, sess, target_eps=None, target_deltas=None):

    assert (target_eps is None) ^ (target_deltas is None)
    eps_deltas = []
    log_moments = sess.run(self._log_moments)
    log_moments_with_order = zip(self._moment_orders, log_moments)
    if target_eps is not None:
      for eps in target_eps:
        eps_deltas.append(
            EpsDelta(eps, self._compute_delta(log_moments_with_order, eps)))
    else:
      assert target_deltas
      for delta in target_deltas:
        eps_deltas.append(
            EpsDelta(self._compute_eps(log_moments_with_order, delta), delta))
    return eps_deltas


class GaussianMomentsAccountant(MomentsAccountant):

  def __init__(self, total_examples, moment_orders=32):

    super(self.__class__, self).__init__(total_examples, moment_orders)
    self._binomial_table = utils.GenerateBinomialTable(self._max_moment_order)

  def _differential_moments(self, sigma, s, t):

    assert t <= self._max_moment_order, ("The order of %d is out "
                                         "of the upper bound %d."
                                         % (t, self._max_moment_order))
    binomial = tf.slice(self._binomial_table, [0, 0],
                        [t + 1, t + 1])
    signs = numpy.zeros((t + 1, t + 1), dtype=numpy.float64)
    for i in range(t + 1):
      for j in range(t + 1):
        signs[i, j] = 1.0 - 2 * ((i - j) % 2)
    exponents = tf.constant([j * (j + 1.0 - 2.0 * s) / (2.0 * sigma * sigma)
                             for j in range(t + 1)], dtype=tf.float64)
    # x[i, j] = binomial[i, j] * signs[i, j] = (i choose j) * (-1)^{i-j}
    x = tf.multiply(binomial, signs)
    # y[i, j] = x[i, j] * exp(exponents[j])
    #         = (i choose j) * (-1)^{i-j} * exp(j(j-1)/(2 sigma^2))
    # Note: this computation is done by broadcasting pointwise multiplication
    # between [t+1, t+1] tensor and [t+1] tensor.
    y = tf.multiply(x, tf.exp(exponents))
    # z[i] = sum_j y[i, j]
    #      = sum_j (i choose j) * (-1)^{i-j} * exp(j(j-1)/(2 sigma^2))
    z = tf.reduce_sum(y, 1)
    return z

  def _compute_log_moment(self, sigma, q, moment_order):

    assert moment_order <= self._max_moment_order, ("The order of %d is out "
                                                    "of the upper bound %d."
                                                    % (moment_order,
                                                       self._max_moment_order))
    binomial_table = tf.slice(self._binomial_table, [moment_order, 0],
                              [1, moment_order + 1])
    # qs = [1 q q^2 ... q^L] = exp([0 1 2 ... L] * log(q))
    qs = tf.exp(tf.constant([i * 1.0 for i in range(moment_order + 1)],
                            dtype=tf.float64) * tf.cast(
                                tf.log(q), dtype=tf.float64))
    moments0 = self._differential_moments(sigma, 0.0, moment_order)
    term0 = tf.reduce_sum(binomial_table * qs * moments0)
    moments1 = self._differential_moments(sigma, 1.0, moment_order)
    term1 = tf.reduce_sum(binomial_table * qs * moments1)
    return tf.squeeze(tf.log(tf.cast(q * term0 + (1.0 - q) * term1,
                                     tf.float64)))
    # 删除所有1的维度

class DummyAccountant(object):
  """An accountant that does no accounting."""

  def accumulate_privacy_spending(self, *unused_args):
    return tf.no_op()

  def get_privacy_spent(self, unused_sess, **unused_kwargs):
    return [EpsDelta(numpy.inf, 1.0)]   