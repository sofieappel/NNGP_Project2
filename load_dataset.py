# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data loader for NNGP experiments.

Loading MNIST dataset with train/valid/test split as numpy array.

PATCHED VERSION: the original file downloaded MNIST via
`tensorflow.examples.tutorials.mnist.input_data`, which points at an
old, now-dead hosting URL and hangs forever with no timeout. This
version fetches MNIST via `tf.keras.datasets.mnist` instead (bundled,
reliable download) and wraps it in a small class that mimics the old
interface so the rest of the file (`_select_mnist_subset`) needs no
changes.

Usage:
mnist_data = load_dataset.load_mnist(num_train=50000, use_float64=True,
                                      mean_subtraction=True)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/nngp/data/',
                     'Directory for data.')


class _Split(object):
  """Mimics the old tf.contrib .train/.validation/.test split object."""

  def __init__(self, images, labels):
    # images: (N, 784) float32 in [0, 1]; labels: (N, 10) one-hot float32
    self.images = images
    self.labels = labels

  @property
  def num_examples(self):
    return self.images.shape[0]


class _Datasets(object):

  def __init__(self, train, validation, test):
    self.train = train
    self.validation = validation
    self.test = test


def _one_hot(labels, num_classes=10):
  out = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
  out[np.arange(labels.shape[0]), labels] = 1.0
  return out


# Edit made by Sofie Appel - 9/24/2024
# Make dataset selectable so MNIST or Fashion MNIST can be used.
_keras_datasets = {
  'mnist': tf.keras.datasets.mnist,
  'fashion_mnist': tf.keras.datasets.fashion_mnist,
}

# Edit made by Sofie Appel - 9/24/2024
def _load_via_keras(dataset='mnist', validation_size=10000):
  """Loads MNIST-shaped Keras datasets through tf.keras (reliable download) and reshapes/
  normalizes/one-hot-encodes it to match the format the rest of this
  file expects from the old input_data.read_data_sets loader."""
  (x_train, y_train), (x_test, y_test) = _keras_datasets[dataset].load_data()

  # Flatten to (N, 784) and scale to [0, 1], matching the old loader.
  x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
  x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0
  y_train = _one_hot(y_train)
  y_test = _one_hot(y_test)

  # Carve out a validation split from the tail of the training set,
  # same as the old loader's validation_size behavior.
  val_images = x_train[-validation_size:]
  val_labels = y_train[-validation_size:]
  train_images = x_train[:-validation_size]
  train_labels = y_train[:-validation_size]

  return _Datasets(
      train=_Split(train_images, train_labels),
      validation=_Split(val_images, val_labels),
      test=_Split(x_test, y_test))


def load_mnist(num_train=50000,
                use_float64=False,
                mean_subtraction=False,
                random_roated_labels=False,
                dataset='mnist'):
  """Loads MNIST (or Fashion-MNIST)as numpy array."""
  datasets = _load_via_keras(dataset,validation_size=10000)
  return _select_mnist_subset(
      datasets,
      num_train,
      use_float64=use_float64,
      mean_subtraction=mean_subtraction,
      random_roated_labels=random_roated_labels)

def load_fashion_mnist(**kwargs):
  return load_mnist(dataset='fashion_mnist', **kwargs)



def _select_mnist_subset(datasets,
                          num_train=100,
                          digits=list(range(10)),
                          seed=9999,
                          sort_by_class=False,
                          use_float64=False,
                          mean_subtraction=False,
                          random_roated_labels=False):
  """Select subset of MNIST and apply preprocessing."""
  np.random.seed(seed)
  digits.sort()
  subset = copy.deepcopy(datasets)
  num_class = len(digits)
  num_per_class = num_train // num_class

  idx_list = np.array([], dtype='uint8')
  ys = np.argmax(subset.train.labels, axis=1)  # undo one-hot

  for digit in digits:
    if datasets.train.num_examples == num_train:
      idx_list = np.concatenate((idx_list, np.where(ys == digit)[0]))
    else:
      idx_list = np.concatenate((idx_list,
                                  np.where(ys == digit)[0][:num_per_class]))
  if not sort_by_class:
    np.random.shuffle(idx_list)

  data_precision = np.float64 if use_float64 else np.float32

  train_image = subset.train.images[idx_list][:num_train].astype(data_precision)
  train_label = subset.train.labels[idx_list][:num_train].astype(data_precision)
  valid_image = subset.validation.images.astype(data_precision)
  valid_label = subset.validation.labels.astype(data_precision)
  test_image = subset.test.images.astype(data_precision)
  test_label = subset.test.labels.astype(data_precision)

  if sort_by_class:
    train_idx = np.argsort(np.argmax(train_label, axis=1))
    train_image = train_image[train_idx]
    train_label = train_label[train_idx]

  if mean_subtraction:
    train_image_mean = np.mean(train_image)
    train_label_mean = np.mean(train_label)
    train_image -= train_image_mean
    train_label -= train_label_mean
    valid_image -= train_image_mean
    valid_label -= train_label_mean
    test_image -= train_image_mean
    test_label -= train_label_mean

  if random_roated_labels:
    r, _ = np.linalg.qr(np.random.rand(10, 10))
    train_label = np.dot(train_label, r)
    valid_label = np.dot(valid_label, r)
    test_label = np.dot(test_label, r)

  return (train_image, train_label,
          valid_image, valid_label,
          test_image, test_label)
