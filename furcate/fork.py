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
from types import FunctionType

import matplotlib.pyplot as plt
import numpy as np

from .config_reader import ConfigReader
from .runner import Runner, seconds_to_string
from .util import get_gpu_stats

logger = logging.getLogger(__name__)


class Fork:
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
        except ValueError as ex:
            logging.basicConfig(
                format="%(asctime)s.%(msecs)06d: %(name)s] %(message)s",
                datefmt=self.date_format,
                level=logging.DEBUG,
            )
            logger.error(ex)

    def is_runner(self):
        """
        Checks if current thread is the runner thread. If there are multiple generated run_configs in ConfigReader
        then thread is classified as runner in order to carry responsibility of ingesting the multiple configs.

        :return: True if runner.
        :rtype: bool
        """
        run_configs, _ = self.config.gen_run_configs()
        return len(run_configs) > 1

    def run(self):
        """
        Starts the model training processes. In situations with multiple configs a new Fork instance will be spawned with
        a specific configuration loadout and data. Further logic is then controlled via the public interface methods.

        The public interface methods will be called in the following order:
            1. `set_visible_gpus`
            2. `get_filepaths`
            3. `get_datasets`
            4. `get_model`
            5. `model_summary` (if configured verbose >= 1)
            6. `get_metrics`
            7. `get_optimizer`
            8. `get_loss`
            9. `model_compile`
            10. `get_callbacks`
            11. `model_fit`
            12. `model_save`
            13. `model_evaluate` (if test dataset present)
            14. `save_metric`
            15. `plot_metric`
            16. `save_history`

        Afterwards a final configuration file with all the run information will be saved in the generated folder.

        :return: None
        """
        self._load_defaults()
        self._load_logging_config()

        if self.is_runner():
            runner = Runner(self.config)
            runner.run(self.script_name, gpu_indices=self.get_available_gpu_indices())
        else:
            self._set_seed()
            self.set_visible_gpus()

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
            run_results["total_time"] = run_time.total_seconds()
            run_results["total_time_string"] = seconds_to_string(
                run_time.total_seconds()
            )

            if test_dataset:
                results = self.model_evaluate(model, test_dataset)
                logger.info("Evaluation results: %s", str(results))
                run_results["results"] = results

            for metric in metrics + ["loss"]:
                self.save_metric(run_results, history, metric)
                self.plot_metric(history, metric)

            with open(os.path.join(self.log_dir, "history.json"), "w") as f:
                self.save_history(history, f)

            with open(os.path.join(self.log_dir, "run_data.json"), "w") as f:
                json.dump(self.data, f)

    def get_available_gpu_indices(self):
        """
        Gets a list of available GPUs indices to train on. Defaults to using nvidia-smi.

        :return: List of available GPUs.
        :rtype: list
        """
        gpus = get_gpu_stats()
        indices = [gpu.id for gpu in gpus]
        return indices

    def set_visible_gpus(self):
        """
        Uses self.gpu_id in order to restrict the training sessions visible GPUs.

        :return: None
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

    def model_summary(self, model):
        """
        Used for displaying the model architecture summary to the user. Called when verbose > 1.

        :param model: Model to be displayed.
        :type model: object
        :return: None
        """
        raise NotImplementedError()

    def get_model(self) -> object:
        """
        Builds the model for use during the training sequence.

        :return: Deep learning model to be compiled and fit.
        :rtype: object
        """
        raise NotImplementedError()

    def get_callbacks(self):
        """
        :return: a list of callbacks to pass to the model during the fit stage. Defaults to None.
        :rtype: list, or None
        """
        return None

    def get_metrics(self) -> list:
        """
        :return: a list of metrics to pass to the model during the compile stage.
        :rtype: list
        """
        raise NotImplementedError()

    def get_optimizer(self) -> FunctionType:
        """
        :return: the model optimizer for use in the compile stage.
        :rtype: object
        """
        raise NotImplementedError()

    def get_loss(self) -> FunctionType:
        """
        :return: the loss function for use in the compile stage.
        :rtype: object
        """
        raise NotImplementedError()

    def model_compile(self, model, optimizer, loss, metrics) -> object:
        """
        Compiles the model for training

        :param model: Model to be compiled
        :type model: object
        :param optimizer: Training optimizer
        :type optimizer: function
        :param loss: Loss function to train against
        :type loss: function
        :param metrics: Metrics to record
        :type metrics: list
        :return: None
        """
        raise NotImplementedError()

    def model_fit(
        self, model, train_set, epochs, valid_set, callbacks, verbose
    ) -> object:
        """
        Trains the compiled model and returns the history of training.

        :param model: Model to train
        :type model: object
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
        raise NotImplementedError()

    def model_evaluate(self, model, test_set):
        """
        Evaluates the model against the provided test set.

        :param model: Model to evaluate
        :type model: object
        :param test_set: Dataset to test the model against
        :type test_set: generator or list
        :return: Evaluation results
        """
        raise NotImplementedError()

    def model_save(self, model):
        """
        Save the model to disk.

        :param model: Trained model to save
        :type model: object
        :return: None
        """
        raise NotImplementedError()

    def preprocess(self, record):
        """
        Preprocesses the data record into appropriate format for the model

        :param record: A single record information to preprocess during dataset mapping for feeding into the model.
        :type record: object
        :return: A preprocessed record to be fed into the model.
        :rtype: object
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
        :rtype: (list, list, list)
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
        :type train_fp: list
        :param test_fp: List of filepaths of test data.
        :type test_fp: list
        :param valid_fp: List of filepaths of validation data.
        :type valid_fp: list
        :return: A generator for each train_set, test_set, valid_set to pass into the model for training.
        """
        raise NotImplementedError()

    def plot(self, metric, epochs, train_metrics, val_metrics):
        """
        Plots the specific metric validation and training metrics against epochs and saves the graph into the configured
        log directory.

        :param metric: String representation of the metric being plotted.
        :type metric: string
        :param epochs: An array of each epoch the model performed [1..N]
        :type epochs: list
        :param train_metrics: Training values at each epoch step.
        :type train_metrics: list
        :param val_metrics: Validation values at each epoch step.
        :type val_metrics: list
        :return: None
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

        :param history (dict): History of model training.
        :param metric (string): Metric to plot.
        :return: None
        """
        raise NotImplementedError()

    def save_metric(self, run_results, history, metric):
        """
        Takes the history object and the provided metric and stores the latest value into the provided dictionary.

        :param run_results: Dictionary to store the last metric value in.
        :type run_results: dict
        :param history: History of model training.
        :type history: dict
        :param metric: Metric to plot.
        :type metric: string
        :return: None
        """
        raise NotImplementedError()

    def save_history(self, history, out_file):
        """
        Saves the model history object to the designated output file.

        :param history: Training history generated during model_fit.
        :type history: dict
        :param out_file: File object to save history to.
        :type out_File: file object
        :return: None
        """
        raise NotImplementedError()
