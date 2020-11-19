# Copyright (c) 2020 Matt Struble. All Rights Reserved.
#
# Use is subject to license terms.
#
# Author: Matt Struble
# Date: Nov. 18 2020
import os
import sys
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from .config_reader import ConfigReader
from .runner import Runner
from .gpu_helper import set_gpus

AUTOTUNE = tf.data.experimental.AUTOTUNE

class Fork(object):
    def __init__(self, config_filename):
        self._load_args()

        if self.is_runner():
            self.config = ConfigReader(config_filename)
        else:
            self.config = ConfigReader(self.args.config_path)

            self.data = self.config.data
            self.meta = self.config.meta_data

    def _load_args(self):
        parser = ArgumentParser
        parser.add_argument('--config', dest='config_path', default=None)
        parser.add_argument('--name', dest='thread_name', default=None)
        parser.add_argument('--gpu', dest='gpu_id', default=None)
        parser.add_argument('--id', dest='thread_id', default=None)

        self.args = parser.parse_args()
        self.script_name = sys.argv[0]

    def _set_visible_gpus(self):
        if self.data['gpu']:
            set_gpus(self.data['gpu'], self.data['framework'])

    def _load_defaults(self):
        self.data.setdefault('log_dir', 'logs')
        self.data.setdefault('learning_rate', 0.001)
        self.data.setdefault('verbose', 2)
        self.data.setdefault('cache', False)
        self.data.setdefault('num_parallel_reads', AUTOTUNE)
        self.data.setdefault('num_parallel_calls', AUTOTUNE)
        self.data.setdefault('seed', None)
        self.data.setdefault('prefetch', 1)

        self.data.setdefault('train_tfrecord', self.data['data_name'] + ".train")
        self.data.setdefault('test_tfrecord', self.data['data_name'] + ".test")
        self.data.setdefault('valid_tfrecord', self.data['data_name'] + ".valid")

    def is_runner(self):
        return self.args.config_path is None and self.args.thread_name is None \
               and self.args.gpu_id is None and self.args.thread_id is None

    def run(self):
        if self.is_runner():
            runner = Runner(self.config)
            runner.run(self.script_name)
        else:
            self._set_visible_gpus()

            train_fp, test_fp, valid_fp = self.get_filepaths()
            train_dataset, test_dataset, valid_dataset = self.get_datasets(train_fp, test_fp, valid_fp)
            model = self.get_model()

            if self.data['verbose'] >= 1:
                model.summary()

            metrics = self.get_metrics()
            optimizer = self.get_optimizer()
            loss = self.get_loss()

            self.model_compile(model, optimizer, loss, metrics)

            callbacks = self.get_callbacks()

            history = self.model_fit(model, train_dataset, self.data['epochs'], valid_dataset, callbacks, self.data['verbose'])

            log_dir = self.data['log_dir']
            model.save(os.path.join(log_dir, 'model.h5'))

            if test_dataset:
                print(self.model_evaluate(model, test_dataset))

            with open(os.path.join(log_dir, 'history.json'), 'w') as f:
                json.dump(history.history, f)

            for metric in metrics:
                self.plot_metric(log_dir, history.history, metric)



    def get_model(self):
        raise NotImplementedError()

    def get_callbacks(self):
        return None

    def get_metrics(self):
        raise NotImplementedError()

    def get_optimizer(self):
        raise NotImplementedError()

    def get_loss(self):
        raise NotImplementedError()

    def model_compile(self, model, optimizer, loss, metrics):
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def model_fit(self, model, train_set, epochs, valid_set, callbacks, verbose):
        history = model.fit(train_set, epochs=epochs, validation_data=valid_set, callbacks=callbacks, verbose=verbose)
        return history

    def model_evaluate(self, model, test_set):
        results = model.evaluate(test_set)
        return results

    def preprocess(self, file):
        raise NotImplementedError()

    def train_preprocess(self, file):
        return self.preprocess(file)

    def test_preprocess(self, file):
        return self.preprocess(file)

    def valid_preprocess(self, file):
        return self.preprocess(file)

    def get_filepaths(self):
        train_filepaths = [os.path.join(self.data['data_dir'], x) for x in os.listdir(self.data['data_dir']) if x.startswith(self.data['train_tfrecord'])]
        test_filepaths = [os.path.join(self.data['data_dir'], x) for x in os.listdir(self.data['data_dir']) if x.startswith(self.data['test_tfrecord'])]
        valid_filepaths = [os.path.join(self.data['data_dir'], x) for x in os.listdir(self.data['data_dir']) if x.startswith(self.data['valid_tfrecord'])]

        return train_filepaths, test_filepaths, valid_filepaths

    def gen_dataset(self, filepaths, processor=preprocess, shuffle_buffer_size=None):
        dataset = tf.data.TFRecordDataset(filepaths, num_parallel_reads=self.data['num_parallel_reads'])

        if self.data['cache']:
            dataset = dataset.cache()
        if shuffle_buffer_size:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=self.data['seed'])

        dataset = dataset.map(processor, num_parallel_calls=self.data['num_parallel_calls']).batch(self.data['batch_size'])

        return dataset.prefetch(self.data['prefetch'])

    def get_datasets(self, train_fp, test_fp, valid_fp):
        train_set = self.gen_dataset(train_fp, shuffle_buffer_size=self.data['shuffle_buffer_size'], processor=self.train_preprocess)
        test_set = self.gen_dataset(test_fp, processor=self.test_preprocess) if len(test_fp) > 0 else None
        valid_set = self.gen_dataset(valid_fp, processor=self.valid_preprocess) if len(valid_fp) > 0 else None

        return train_set, test_set, valid_set

    def plot_metric(self, log_dir, history, metric):
        if not isinstance(metric, str):
            try:
                metric = metric.name
            except:
                return

        if metric not in history:
            return

        train_metrics = history[metric]
        val_metrics = history['val_' + metric]
        epochs = range(1, len(train_metrics) + 1)

        plt.plot(epochs, train_metrics)
        plt.plot(epochs, val_metrics)
        # plt.gca().set_ylim(0,-1)# sets the vertical range within [0, -1]
        plt.title('Training and Validation ' + metric)
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend(["train_" + metric, 'val_' + metric])
        plt.savefig(os.path.join(log_dir, metric + '.jpg'), bbox_inches='tight', dpi=150)
        plt.clf()

