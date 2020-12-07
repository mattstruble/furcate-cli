# Copyright (c) 2020 Matt Struble. All Rights Reserved.
#
# Use is subject to license terms.
#
# Author: Matt Struble
# Date: Nov. 18 2020
import json
import logging
import logging.config
import os
import random
import sys
from argparse import ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .config_reader import ConfigReader
from .gpu_helper import set_gpus
from .runner import Runner, seconds_to_string

logger = logging.getLogger(__name__)


class Fork(object):
    date_format = "%Y-%m-%d %H:%M:%S"

    def __init__(self, config_filename):
        self._load_args()

        if self.args.config_path:
            self.config = ConfigReader(self.args.config_path)
        else:
            self.config = ConfigReader(config_filename)

    @property
    def data(self):
        return self.config.data

    @property
    def meta(self):
        return self.data["meta"]

    @property
    def log_dir(self):
        return self.data["log_dir"]

    @property
    def learning_rate(self):
        return self.data["learning_rate"]

    @property
    def verbose(self):
        return self.data["verbose"]

    @property
    def cache(self):
        return self.data["cache"]

    @property
    def seed(self):
        return self.data["seed"]

    @property
    def prefetch(self):
        return self.data["prefetch"]

    @property
    def gpu_id(self):
        return self.data["gpu_id"]

    def __getattr__(self, item):
        return self.data[item]

    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

    def _load_args(self):
        parser = ArgumentParser()
        parser.add_argument("--config", dest="config_path", default=None)
        parser.add_argument("--name", dest="thread_name", default=None)
        parser.add_argument("--gpu", dest="gpu_id", default=-1)
        parser.add_argument("--id", dest="thread_id", default=None)
        parser.add_argument("--log_config", dest="log_config", default=None)

        self.args = parser.parse_args()
        self.script_name = sys.argv[0]

    def _set_visible_gpus(self):
        if self.gpu_id > -1:
            set_gpus(self.gpu_id, self.meta["framework"])

    def _load_defaults(self):
        self.data.setdefault("gpu_id", self.args.gpu_id)

    def _load_logging_config(self):
        if not self.args.log_config:
            log_fname = os.path.join(
                os.path.abspath(os.path.dirname(__file__)), "config", "logging.config"
            )
        else:
            log_fname = self.args.log_config

        try:
            with open(log_fname, "r") as f:
                data = json.load(f)
            logging.config.dictConfig(data)
        except Exception as e:
            logging.basicConfig(
                format="%(asctime)s.%(msecs)06d: %(name)s] %(message)s",
                datefmt=self.date_format,
                level=logging.DEBUG,
            )
            logger.warning(e)

    def is_runner(self):
        run_configs, _ = self.config.gen_run_configs()
        return len(run_configs) > 1

    def run(self):
        self._load_defaults()
        self._load_logging_config()

        if self.is_runner():
            runner = Runner(self.config)
            runner.run(self.script_name)
        else:
            self._set_seed()
            self._set_visible_gpus()

            self.meta["data"] = {}
            run_results = self.meta["data"]

            start_time = datetime.now()
            run_results["start_time"] = start_time.timestamp()
            run_results["start_time_string"] = start_time.strftime(self.date_format)

            train_fp, test_fp, valid_fp = self.get_filepaths()
            train_dataset, test_dataset, valid_dataset = self.get_datasets(
                train_fp, test_fp, valid_fp
            )
            model = self.get_model()

            if self.verbose >= 1:
                self.model_summary(model)

            metrics = self.get_metrics()
            optimizer = self.get_optimizer()
            loss = self.get_loss()

            self.model_compile(model, optimizer, loss, metrics)

            callbacks = self.get_callbacks()

            history = self.model_fit(
                model,
                train_dataset,
                self.epochs,
                valid_dataset,
                callbacks,
                self.verbose,
            )

            self.model_save(model)

            end_time = datetime.now()
            run_results["end_time"] = end_time.timestamp()
            run_results["end_time_string"] = end_time.strftime(self.date_format)

            run_time = end_time - start_time
            run_results["run_time"] = run_time.total_seconds()
            run_results["run_time_string"] = seconds_to_string(run_time.total_seconds())

            if test_dataset:
                results = self.model_evaluate(model, test_dataset)
                logger.info("Evaluation results: %s", str(results))
                run_results["results"] = results

            for metric in metrics + ["loss"]:
                self.save_metric(run_results, history, metric)
                self.plot_metric(history, metric)

            with open(os.path.join(self.log_dir, "history.json"), "w") as f:
                json.dump(history.history, f)

            with open(os.path.join(self.log_dir, "run_data.json"), "w") as f:
                json.dump(self.data, f)

    def model_summary(self, model):
        """
        Used for displaying the model architecture summary to the user. Called when verbose > 1.
        :param model: Model to be displayed.
        :return: None
        """
        raise NotImplementedError()

    def get_model(self) -> object:
        """
        Builds the model for use during the training sequence.
        :return: Deep learning model to be compiled and fit.
        """
        raise NotImplementedError()

    def get_callbacks(self):
        """
        Returns a list of callbacks to pass to the model during the fit stage. Defaults to None.
        """
        return None

    def get_metrics(self) -> list:
        """
        Returns a list of metrics to pass to the model during the compile stage.
        """
        raise NotImplementedError()

    def get_optimizer(self) -> object:
        """
        Returns the model optimizer for use in the compile stage.
        """
        raise NotImplementedError()

    def get_loss(self) -> object:
        """
        Returns the loss function for use in the compile stage.
        """
        raise NotImplementedError()

    def model_compile(self, model, optimizer, loss, metrics) -> object:
        """
        Compiles the model for training
        :param model: Model to be compiled
        :param optimizer: Training optimizer
        :param loss: Loss function to train against
        :param metrics: Metrics to record
        :return: Compiled model
        """
        raise NotImplementedError()

    def model_fit(
        self, model, train_set, epochs, valid_set, callbacks, verbose
    ) -> object:
        """
        Trains the compiled model and returns the history of training.
        :param model: Model to train
        :param train_set: Training dataset to fit against
        :param epochs: Number of epochs to train
        :param valid_set: Validation dataset
        :param callbacks: A list of training callbacks
        :param verbose: Verbosity of training
        :return: History object representing the model training
        """
        raise NotImplementedError()

    def model_evaluate(self, model, test_set):
        """
        Evaluates the model against the provided test set.
        :param model: Model to evaluate
        :param test_set: Dataset to test the model against
        :return: Evaluation results
        """
        raise NotImplementedError()

    def model_save(self, model):
        """
        Save the model to disk.
        :param model: Trained model to save
        :return: None
        """
        raise NotImplementedError()

    def preprocess(self, record):
        """
        Preprocesses the data record into appropriate format for the model
        :param record: Record information to preprocess
        :return: Preprocessed dataset for model ingestion
        """
        raise NotImplementedError()

    def train_preprocess(self, record):
        """
        Preprocessor specifically for training records. Defaults to preprocess.
        """
        return self.preprocess(record)

    def test_preprocess(self, record):
        """
        Preprocessor specifically for test records. Defaults to preprocess.
        """
        return self.preprocess(record)

    def valid_preprocess(self, record):
        """
        Preprocessor specifically for validation records. Defaults to preprocess.
        """
        return self.preprocess(record)

    def get_filepaths(self):
        """
        Gets the filepaths to the data that will then be processed by get_dataset.
        :return: train_filepaths, test_filepaths, valid_filepaths
        """
        train_filepaths = [
            os.path.join(self.data_dir, x)
            for x in os.listdir(self.data_dir)
            if x.startswith(self.train_prefix)
        ]
        test_filepaths = [
            os.path.join(self.data_dir, x)
            for x in os.listdir(self.data_dir)
            if x.startswith(self.test_prefix)
        ]
        valid_filepaths = [
            os.path.join(self.data_dir, x)
            for x in os.listdir(self.data_dir)
            if x.startswith(self.valid_prefix)
        ]

        return train_filepaths, test_filepaths, valid_filepaths

    def get_datasets(self, train_fp, test_fp, valid_fp):
        """
        Gets the datasets to be passed into the model for training and evaluation.
        :param train_fp: List of filepaths of training data.
        :param test_fp: List of filepaths of test data.
        :param valid_fp: List of filepaths of validation data.
        :return: train_set, test_set, valid_set.
        """
        raise NotImplementedError()

    def plot(self, metric, epochs, train_metrics, val_metrics):
        """
        Plots the specific metric validation and training metrics against epochs and saves the graph into the configured
        log directory.

        :param metric: String representation of the metric being plotted.
        :param epochs: An array of each epoch the model performed [1..N]
        :param train_metrics: Training values at each epoch step.
        :param val_metrics: Validation values at each epoch step.
        """
        plt.plot(epochs, train_metrics)
        plt.plot(epochs, val_metrics)
        # plt.gca().set_ylim(0,-1)# sets the vertical range within [0, -1]
        plt.title("Training and Validation " + metric)
        plt.xlabel("Epochs")
        plt.ylabel(metric.capitalize())
        plt.legend(["train_" + metric.lower(), "val_" + metric.lower()])
        plt.savefig(
            os.path.join(self.log_dir, metric + ".jpg"), bbox_inches="tight", dpi=150
        )
        plt.clf()

    def plot_metric(self, history, metric):
        """
        Takes the history object and the provided metric and graphs them using plot.
        :param history: History of model training.
        :param metric: Metric to plot.
        """
        raise NotImplementedError()

    def save_metric(self, dict, history, metric):
        """
        Takes the history object and the provided metric and stores the latest value into the provided dictionary.
        :param dict: Dictionary to store the last metric value in.
        :param history: History of model training.
        :param metric: Metric to plot.
        """
        raise NotImplementedError()


class ForkTF(Fork):
    import tensorflow as tf

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
