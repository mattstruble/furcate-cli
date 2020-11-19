# Copyright (c) 2020 Matt Struble. All Rights Reserved.
#
# Use is subject to license terms.
#
# Author: Matt Struble
# Date: Nov. 18 2020
import json
import copy

class ConfigReader(object):
    # Bare minimum configuration keys required to run a default training fork
    _REQUIRED_KEYS = ['data_name', 'data_dir', 'batch_size', 'epochs']

    def __init__(self, filename):
        self.data, self.meta_data = self._load_config(filename)
        self._load_defaults()

        self.run_configs = []

    def gen_run_configs(self):
        if len(self.run_configs) == 0:
            self._gen_run_configs(self.data)

        return self.run_configs

    def _load_defaults(self):
        self.data.setdefault('log_dir', 'logs')
        self.data.setdefault('learning_rate', 0.001)
        self.data.setdefault('verbose', 2)
        self.data.setdefault('cache', False)
        self.data.setdefault('seed', 42)
        self.data.setdefault('prefetch', 1)

        self.data.setdefault('train_tfrecord', self.data['data_name'] + ".train")
        self.data.setdefault('test_tfrecord', self.data['data_name'] + ".test")
        self.data.setdefault('valid_tfrecord', self.data['data_name'] + ".valid")

    def _load_config(self, config):
        with open(config) as f:
            data = json.load(f)

        for key in self._REQUIRED_KEYS:
            if key not in data.keys():
                raise ValueError("The configuration file '{}' does not contain the required key: {}".format(config, key))

        meta_data = data.pop('meta', None)

        return data, meta_data

    def _gen_config_permutations(self, index, dict, enumerated_data):
        if index >= len(enumerated_data):
            return [dict]

        key = enumerated_data[index]['key']

        if type(enumerated_data[index]['value']) is list:
            values = enumerated_data[index]['value']
        else:
            values = [enumerated_data[index]['value']]

        result = []
        for value in values:
            tmp = copy.deepcopy(dict)
            tmp[key] = value
            result.extend(self._gen_config_permutations(index + 1, tmp, enumerated_data))

        return result

    def _gen_run_configs(self, data):
        enumerated_data = {}

        for index, (key, value) in enumerate(data.items()):
            enumerated_data[index] = {
                'key': key,
                'value': value
            }

        self.run_configs = self._gen_config_permutations(0, {}, enumerated_data)

        for config in self.run_configs:
            config['meta'] = copy.deepcopy(self.meta_data)