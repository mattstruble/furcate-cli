# Copyright (c) 2020 Matt Struble. All Rights Reserved.
#
# Use is subject to license terms.
#
# Author: Matt Struble
# Date: Nov. 18 2020
import json
import copy
import random

class ConfigReader(object):
    # Bare minimum configuration keys required to run a default training fork
    _REQUIRED_KEYS = ['data_name', 'data_dir', 'batch_size', 'epochs']

    def __init__(self, filename):
        self.data, self.meta_data = self._load_config(filename)
        self._load_defaults()

        self.run_configs = []
        self.permutable_keys = set()

    def gen_run_configs(self):
        if len(self.run_configs) == 0:
            self._gen_run_configs(self.data)
            self._clean_configs()
            random.shuffle(self.run_configs)

        return self.run_configs, self.permutable_keys

    def _load_defaults(self):
        self.data.setdefault('log_dir', 'logs')

        self.data.setdefault('learning_rate', 0.001)
        self.data.setdefault('verbose', 2)
        self.data.setdefault('cache', False)
        self.data.setdefault('seed', 42)
        self.data.setdefault('prefetch', 1)

        self.meta_data.setdefault('allow_cpu', False)
        self.meta_data.setdefault('exclude_configs', [])

        self.data.setdefault('train_prefix', self.data['data_name'] + ".train")
        self.data.setdefault('test_prefix', self.data['data_name'] + ".test")
        self.data.setdefault('valid_prefix', self.data['data_name'] + ".valid")

    def _load_config(self, config):
        with open(config) as f:
            data = json.load(f)

        for key in self._REQUIRED_KEYS:
            if key not in data.keys():
                raise ValueError("The configuration file '{}' does not contain the required key: {}".format(config, key))

        data.setdefault('meta', {})

        return data, data['meta']

    def _gen_config_permutations(self, index, dict, enumerated_data):
        if index >= len(enumerated_data):
            return [dict]

        key = enumerated_data[index]['key']

        if type(enumerated_data[index]['value']) is list:
            self.permutable_keys.add(key)
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

    def _clean_configs(self):
        skip_configs = self.meta_data['exclude_configs']
        if len(skip_configs) > 0:
            to_remove = []
            for i in range(len(self.run_configs)):
                run_config = self.run_configs[i]
                remove = True
                for config in skip_configs:
                    for key, value in config.items():
                        if run_config[key] != value:
                            remove = False
                            break

                if remove:
                    to_remove.append(i)

            for idx in to_remove:
                del self.run_configs[idx]