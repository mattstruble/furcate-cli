# Copyright (c) 2020 Matt Struble. All Rights Reserved.
#
# Use is subject to license terms.
#
# Author: Matt Struble
# Date: Nov. 18 2020
import os
import tempfile
import subprocess
import threading
import time
import json

from .gpu_helper import get_gpus

class TrainingThread (threading.Thread):

    def __init__(self, id, config, script_name):
        threading.Thread.__init__(self)
        self.threadID = id
        self.config = config
        self.script_name = script_name

        self.dir_name = os.path.basename(self.config['data_dir'])
        self.name = self.dir_name + str(id)

    def _gen_log_dir(self):
        folder = "{}_{}".format(self.name, self.dir_name)

        for key in self.config.keys():
            if key not in self.config['meta']['ignore_log']:
                short_key = ''.join([s[0] for s in key.split('_')])

                if self.config[key] is not None:
                    value = str(self.config[key]).replace('.','-')
                    folder += "_{}{}".format(short_key, value)

        self.config['log_dir'] = os.path.join(self.config['log_dir'], folder)

        if not os.path.exists(self.config['log_dir']):
            os.makedirs(self.config['log_dir'])

    def _generate_run_command(self, config_path):
        command = 'python3 {} --config "{}" --name "{}" --id "{}"'.format(
            self.script_name, config_path, self.name, self.threadID)

        if self.config['gpu']:
            command += ' --gpu "{}"'.format(self.config['gpu'])

        return command

    def run(self):

        fd, temppath = tempfile.mkstemp()

        try:
            self._gen_log_dir()

            with os.fdopen(fd, 'w') as tmp:
                json.dump(self.config, tmp)

            command = self._generate_run_command(temppath)

            print('Starting:', os.getcwd(), command)

            with open(os.path.join(self.config['log_dir'], self.name + '.log'), 'w', encoding='utf-8') as log, \
                open(os.path.join(self.config['log_dir'], self.name + '.err'), 'w', encoding='utf-8') as err:
                subprocess.call(command, shell=True, stdout=log, stderr=err)

        finally:
            os.remove(temppath)


class Runner(object):

    def __init__(self, config):
        self.config = config
        self.meta = config.meta_data

        self.run_configs = self.config.gen_run_configs()

    def run(self, script_name, framework='tf'):
        gpus = get_gpus(framework)

        if len(gpus) < 1 and ('allow_cpu' not in self.meta or self.meta['allow_cpu'] is False):
            raise ValueError(
                "CPU processing is not enabled and could not find GPU devices to run on. If you want to enable CPU processing please update the config: { 'meta': { 'allow_cpu': true } }")

        max_threads = self._get_max_threads(gpus)

        main_thread = threading.current_thread()
        thread_id = 0
        gpu_mapping = {}

        if max_threads > 1:
            gpu_idxs = list(range(len(gpus)))

        while len(self.run_configs) > 0:
            while threading.activeCount() -1 == max_threads:
                time.sleep(60)

            if max_threads > 1:
                to_del = []
                for t, gpu in gpu_mapping.items():
                    if not t.isAlive():
                        gpu_idxs.append(gpu)
                        to_del.append(t)

                for t in to_del:
                    del gpu_mapping[t]

                gpu = gpu_idxs.pop()
            else:
                gpu = None

            config = self.run_configs.pop()
            config['gpu'] = gpu
            config['framework'] = framework

            training = TrainingThread(thread_id, config, script_name)
            training.start()

            gpu_mapping[training] = gpu
            thread_id += 1

        for t in threading.enumerate():
            if t is not main_thread:
                t.join()


    def _get_max_threads(self, gpus):
        if self.meta and 'max_threads' in self.meta:
            max_threads = min(1, self.meta['max_threads'])

            if max_threads > len(gpus) > 1:
                print(
                    "Configured max_threads [{}] is higher than total number of GPUs [{}]. Defaulting to number of GPUs".format(
                        max_threads, len(gpus)))
                max_threads = min(len(gpus), max_threads)
        else:
            max_threads = max(1, len(gpus))
            print("Couldn't find max_threads in config, defaulting to [{}].".format(max_threads))

        if len(gpus) > max_threads:
            print(
                "Potentially not utilizing all the GPUs. Check the config to ensure the meta tag 'max_threads' is set properly: { 'meta': { 'max_threads': X } }")

        return max_threads

