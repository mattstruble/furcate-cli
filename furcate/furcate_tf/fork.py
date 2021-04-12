# Copyright (c) 2020 Matt Struble. All Rights Reserved.
#
# Use is subject to license terms.
#
# Author: Matt Struble
# Date: Dec. 07 2020
import json
import os
import re

import tensorflow as tf

import furcate.fork


class Fork(furcate.fork.Fork):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    @property
    def num_parallel_reads(self):
        return self.data["num_parallel_reads"]

    @property
    def num_parallel_calls(self):
        return self.data["num_parallel_calls"]

    def _load_defaults(self):
        """
        Sets num_parallel_reads and num_parallel_calls to tf.AUTOTUNE, and sets framework to tf.
        """
        super()._load_defaults()

        self.data.setdefault("num_parallel_reads", self.AUTOTUNE)
        self.data.setdefault("num_parallel_calls", self.AUTOTUNE)
        self.config.meta_data.setdefault("framework", "tf")

    def _set_seed(self):
        super()._set_seed()
        tf.random.set_seed(self.seed)

    def get_available_gpu_indices(self):
        """
        Gets a list of available GPUs indices to train on using `tf.config.list_physical_devices`

        :return: List of available GPUs
        :rtype: list
        """
        gpus = tf.config.list_physical_devices("GPU")
        indices = [int(re.findall(r"\d+", gpu.name)[0]) for gpu in gpus]
        return indices

    def set_visible_gpus(self):
        """
        Uses `self.gpu_id` to set the trainign session's visible GPUs using `tf.config.set_visible_devices`

        :return: None
        """
        if self.gpu_id > -1:
            gpus = tf.config.list_physical_devices("GPU")
            visible_devices = [gpu for gpu in gpus if str(self.gpu_id) in gpu.name]
            tf.config.set_visible_devices(visible_devices, "GPU")

    def model_compile(self, model, optimizer, loss, metrics):
        """
        Compiles the model for training.

        :param model: Model to be compiled
        :type model: tf.keras.Model
        :param optimizer: Training optimizer
        :type optimizer: function
        :param loss: Loss function to train against
        :type loss: function
        :param metrics: Metrics to record
        :type metrics: list
        :return: None
        """
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def model_summary(self, model):
        """
        Used for displaying the model architecture summary to the user. Called when verbose > 1.

        :param model: Model to be displayed.
        :type model: tf.keras.Model
        :return: None
        """
        model.summary()

    def model_fit(self, model, train_set, epochs, valid_set, callbacks, verbose):
        """
        Trains the compiled model and returns the history of training.

        :param model: Model to train
        :type model: tf.keras.Model
        :param train_set: Training dataset to fit against
        :type train_set: generator or list
        :param epochs: Number of epochs to train
        :type epochs: int
        :param valid_set: Validation dataset
        :type valid_set: generator or list
        :param callbacks: A list of training callbacks
        :type callbacks: list
        :param verbose: Verbosity of training
        :type verbose: int
        :return: History object representing the model training
        :rtype: dict
        """
        history = model.fit(
            train_set,
            epochs=epochs,
            validation_data=valid_set,
            callbacks=callbacks,
            verbose=verbose,
        )
        return history

    def model_evaluate(self, model, test_set):
        """
        Evaluates the model against the provided test set.

        :param model: Model to evaluate
        :type model: tf.keras.Model
        :param test_set: Dataset to test the model against
        :type test_set: generator or list
        :return: Evaluation results
        """
        results = model.evaluate(test_set)
        return results

    def model_save(self, model):
        """
        Save the model to disk.

        :param model: Trained model to save
        :type model: tf.keras.Model
        :return: None
        """
        model.save(os.path.join(self.log_dir, "model.h5"))

    def gen_tfrecord_dataset(self, filepaths, processor, shuffle=False):
        """
        Generates a `TFRecordDataset` mapped via the passed in processor.

        :param filepaths: list of filepaths
        :type filepaths: list
        :param processor: Function to map the dataset against
        :type processor: function
        :param shuffle: whether to shuffle the dataset
        :type shuffle: bool
        :return: A dataset that has been mapped, cached, batched, and prefetched.
        :rtype: tf.data.TFRecordDataset
        """
        dataset = tf.data.TFRecordDataset(
            filepaths, num_parallel_reads=self.num_parallel_reads
        )

        if self.cache:
            dataset = dataset.cache()
        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=self.shuffle_buffer_size, seed=self.seed
            )

        dataset = dataset.map(
            processor, num_parallel_calls=self.num_parallel_calls
        ).batch(self.batch_size)

        return dataset.prefetch(self.prefetch)

    def _get_metric(self, metric, history):
        if not isinstance(metric, str):
            try:
                metric = metric.name
            except NameError:
                return None

        if metric not in history:
            return None

        return metric

    def plot_metric(self, history, metric):
        metric_name = self._get_metric(metric, history.history)
        if metric_name is None:
            return

        train_metrics = history.history[metric_name]
        val_metrics = history.history["val_" + metric_name]
        epochs = range(1, len(train_metrics) + 1)

        self.plot(metric_name, epochs, train_metrics, val_metrics)

    def save_metric(self, run_results, history, metric):
        metric_name = self._get_metric(metric, history.history)
        if metric_name is None:
            return

        train_metrics = history.history[metric_name]
        val_metrics = history.history["val_" + metric_name]

        run_results["train_" + metric_name] = train_metrics[-1]
        run_results["val_" + metric_name] = val_metrics[-1]
        run_results["epochs"] = len(val_metrics)

    def save_history(self, history, out_file):
        json.dump(history.history, out_file)
