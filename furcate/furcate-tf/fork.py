# Copyright (c) 2020 Matt Struble. All Rights Reserved.
#
# Use is subject to license terms.
#
# Author: Matt Struble
# Date: Dec. 07 2020
import os
import re
import furcate.fork
import tensorflow as tf


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
        gpus = tf.config.list_physical_devices("GPU")
        indices =  [int(re.findall(r"\d+", gpu.name)[0]) for gpu in gpus]
        return indices

    def set_visible_gpus(self):
        if self.gpu_id > -1:
            gpus = tf.config.list_physical_devices("GPU")
            visible_devices = [gpu for gpu in gpus if str(self.gpu_id) in gpu.name]
            tf.config.set_visible_devices(visible_devices, "GPU")

    def model_compile(self, model, optimizer, loss, metrics):
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def model_summary(self, model):
        model.summary()

    def model_fit(self, model, train_set, epochs, valid_set, callbacks, verbose):
        history = model.fit(
            train_set,
            epochs=epochs,
            validation_data=valid_set,
            callbacks=callbacks,
            verbose=verbose,
        )
        return history

    def model_evaluate(self, model, test_set):
        results = model.evaluate(test_set)
        return results

    def model_save(self, model):
        model.save(os.path.join(self.log_dir, "model.h5"))

    def gen_tfrecord_dataset(self, filepaths, processor, shuffle=False):
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

    def save_metric(self, dict, history, metric):
        metric_name = self._get_metric(metric, history.history)
        if metric_name is None:
            return

        train_metrics = history.history[metric_name]
        val_metrics = history.history["val_" + metric_name]

        dict["train_" + metric_name] = train_metrics[-1]
        dict["val_" + metric_name] = val_metrics[-1]
        dict["epochs"] = len(val_metrics)
